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
from .topology import build_circuit
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
    max_dim: int = WORK_MAX_DIM,
    verify: bool = False,
    bridge_px: int | None = None,
    method: str = "flatfield",
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
    graph, node_map = build_circuit(img, dets, values=values, bridge_px=bridge_px, method=method)

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

            res.verification = verify_graph(graph, asc)

    return res
