"""End-to-end: photograph of a hand-drawn circuit -> LTspice schematic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .asc_writer import write_asc
from .detect import detections_from_voc, draw, nms_across_classes, scale_detections
from .graph import CircuitGraph
from .netlist import DEFAULT_ANALYSIS, to_netlist
from .topology import add_supplies, build_circuit, reclassify, set_pins
from .wires import NodeMap, colorize

#: Wire tracing runs at this resolution.  CGHD photos are often 3000px+ on the
#: long side, where skeletonisation is slow and no more accurate.
WORK_MAX_DIM = 1600


@dataclass
class Result:
    image_path: Path
    graph: CircuitGraph
    node_map: NodeMap
    detections: list
    work_image: np.ndarray
    outputs: dict = field(default_factory=dict)
    verification: object | None = None


def _load_and_scale(image_path: Path, dets: list, max_dim: int):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image {image_path}")
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        dets = scale_detections(dets, scale, scale)
    return img, dets


def process(
    image_path: str | Path,
    *,
    detector=None,
    gt_xml: str | Path | None = None,
    out_dir: str | Path | None = None,
    analysis: str = DEFAULT_ANALYSIS,
    values: dict | None = None,
    nets: dict | None = None,
    flip: list[str] | None = None,
    pins: dict | None = None,
    reclass: dict | None = None,
    supplies: dict | None = None,
    read_values: bool = True,
    ocr_mode: str = "auto",
    max_dim: int = WORK_MAX_DIM,
    verify: bool = False,
    bridge_px: int | None = None,
    method: str = "flatfield",
    unet_thr: float | None = None,
    stem: str | None = None,
) -> Result:
    """Run the full pipeline on one image."""
    image_path = Path(image_path)
    if detector is None and gt_xml is None:
        raise ValueError("pass either a detector or gt_xml")

    if gt_xml is not None:
        dets = detections_from_voc(gt_xml)
    else:
        dets = nms_across_classes(detector(str(image_path)))

    img, dets = _load_and_scale(image_path, dets, max_dim)
    # ``values`` is deliberately *not* passed to build_circuit: its refs are
    # positional there, and the drawing's own designators permute them a few
    # lines below. Explicit values are applied once, at the end, against the
    # refs the netlist actually prints.
    graph, node_map = build_circuit(img, dets, bridge_px=bridge_px,
                                    method=method, unet_thr=unet_thr)
    value_read: set = set()

    # Read the handwritten values, where the labels can be understood. Explicit
    # --value overrides take precedence, so correcting the pipeline by hand
    # still works.
    if read_values:
        from .values import (
            apply_names_to_graph, apply_to_graph, labels_from_detections,
            name_nets_from_labels, read_text_boxes, values_from_detections,
        )

        # Ground-truth annotations already carry the strings; with the detector
        # there is nothing to read them from unless a recogniser is installed.
        if not any(d.cls == "text" and getattr(d, "text", "") for d in dets):
            # Closed-vocabulary scoring first: it cannot return an unparseable
            # string, because every candidate it ranks is already a legal
            # label. Free-running OCR is the fallback for a drawing that does
            # not follow a naming convention.
            got = 0
            if ocr_mode in ("auto", "structured"):
                try:
                    from .ocr import read_structured

                    info: dict = {}
                    got = read_structured(img, dets, info=info)
                    if got and info.get("reader"):
                        graph.warnings.append(
                            f"labels read by {info['reader']}")
                    # A reader split between GPU and CPU is right and slow,
                    # which is the failure mode nobody reports as a failure.
                    if info.get("slow"):
                        graph.warnings.append(info["slow"])
                except Exception as exc:
                    # A recogniser that failed to load reads nothing, which
                    # looks exactly like illegible handwriting unless it is
                    # said out loud. It cost an afternoon once already.
                    graph.warnings.append(f"structured OCR unavailable: {exc}")
                    got = 0
            if not got and ocr_mode in ("auto", "free"):
                read_text_boxes(img, dets)

        # Only now can a ground symbol be checked against what the drawing
        # calls it, because only now has anything been read. A rail marker
        # taken for a ground is the one detector error that is not local --
        # every ground is merged onto one node -- so it is worth rebuilding
        # the graph for. That is cheap: the node map is already traced and
        # handed straight back, so this is ring statistics again, not
        # segmentation and skeletonisation again.
        from .values import retag_rail_grounds

        retagged = retag_rail_grounds(dets)
        if retagged:
            graph, node_map = build_circuit(
                img, dets, node_map=node_map, bridge_px=bridge_px,
                method=method, unet_thr=unet_thr)
            graph.warnings.append(
                f"re-typed {retagged} ground symbol(s) the drawing labels as "
                f"a rail, and rebuilt the nets")

        n = apply_to_graph(graph, values_from_detections(dets),
                           applied=value_read)
        if n:
            graph.warnings.append(f"read {n} component value(s) from the drawing")
        # Rename to the designators on the page. Ours are positional, so our
        # "R1" is the topmost resistor and almost never the drafter's R1,
        # which makes the netlist impossible to read against the drawing.
        r = apply_names_to_graph(graph, labels_from_detections(dets))
        if r:
            graph.warnings.append(f"took {r} component name(s) from the drawing")
        # Name the ports, so the result can be probed as V(Vout) rather than
        # by guessing which N00n the output turned out to be.
        k = name_nets_from_labels(graph, dets, overrides=nets)
        if k:
            graph.warnings.append(f"named {k} net(s) from the drawing")

    # Polarity the drawing does not give up on its own. Diode orientation is
    # now read by a learned model (``diode_net``, 1.000 out-of-fold over 5145
    # CGHD diodes), but it declines rather than guesses when the ink does not
    # say, and source polarity is still unread, so the exact override stays.
    # Re-typing comes *after* the designators are taken off the page, because
    # the ref it is given is the ref the user is looking at. Ours are
    # positional until `apply_names_to_graph` runs, and the drawing's own
    # names routinely permute them -- the V2 in the netlist is not the second
    # source down the page. An override keyed on an invisible intermediate
    # name would be unusable.
    reclassify(graph, reclass or {})
    # A re-typed part can finally take its name *and its value* off the page.
    # Both were refused before: a rename cannot change the SPICE prefix, and a
    # label is only offered to a component whose class could carry it, so the
    # drawing's "L1" and "100u" had nowhere to go while the part was a
    # resistor. Left out, the two inductors of a boost converter come back as
    # L1 and L2 the other way round from the drawing, each holding the other's
    # henries -- which is harmless electrically and makes the netlist
    # impossible to check against the page by eye.
    if reclass and read_values:
        from .values import (
            apply_names_to_graph, apply_to_graph, labels_from_detections,
            values_from_detections,
        )

        from .ocr import rerepair

        rerepair(dets)        # the repair was unit-dependent; the unit moved
        apply_to_graph(graph, values_from_detections(dets), applied=value_read)
        apply_names_to_graph(graph, labels_from_detections(dets))
    # An explicit value is applied last, since the part it names may only have
    # acquired that name a moment ago.
    for _ref, _v in (values or {}).items():
        for _c in graph.components:
            if _c.ref.upper() == _ref.upper():
                _c.value = _v

    # After naming, so a supply can be asked for by the name on the drawing
    # (``--supply VCC=5``) rather than by whichever N00n it happened to get.
    add_supplies(graph, supplies or {})

    # Say which parts were guessed. A default is invisible in the netlist --
    # `C2 N003 0 1u` does not record whether the 1u was read or assumed -- so
    # a value the reader missed looks exactly like one it got right.
    from .values import defaulted_components

    guessed = defaulted_components(graph, read_ids=value_read,
                                   explicit=set(values or ()))
    if guessed:
        graph.warnings.append(
            f"{len(guessed)} part(s) kept the default value, not read from "
            f"the drawing: {', '.join(guessed)}")

    # Last, so a pin order can be typed using the net names the
    # netlist prints -- which only exist once naming has run.
    set_pins(graph, pins or {})

    for ref in (flip or ()):
        for c in graph.components:
            if c.ref.upper() == ref.upper():
                c.nodes = list(reversed(c.nodes))
                graph.warnings.append(f"reversed {c.ref} by hand")

    res = Result(image_path=image_path, graph=graph, node_map=node_map,
                 detections=dets, work_image=img)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = stem or image_path.stem
        asc = out_dir / f"{name}.asc"
        write_asc(graph, asc, analysis=analysis)
        cir = out_dir / f"{name}.cir"
        cir.write_text(to_netlist(graph, title=name, analysis=analysis))
        js = out_dir / f"{name}.netlist.json"
        js.write_text(graph.to_json())
        ov = out_dir / f"{name}.detections.png"
        cv2.imwrite(str(ov), draw(img, dets))
        nm = out_dir / f"{name}.nodes.png"
        cv2.imwrite(str(nm), colorize(node_map))
        res.outputs = {"asc": asc, "cir": cir, "json": js,
                       "detections_png": ov, "nodes_png": nm}

        if verify:
            from .verify import verify_graph

            # Pass the analysis through: verify_graph rewrites the .asc, and
            # dropping it here would replace the caller's directive with the
            # default, so --verify would quietly disable --analysis.
            res.verification = verify_graph(graph, asc, analysis=analysis)

    return res
