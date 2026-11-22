"""Score the component values read from the drawing against CGHD's references.

Values used to be defaults -- R=1k, C=1u, L=1m -- correctable only by hand with
``--value R1=4.7k``. :mod:`circuit2spice.values` reads them from the handwritten
labels instead. This measures how much that recovers.

Two scores, because the reference is not always a fair target:

``all``
    Every component in the reference netlist.

``informative`` (the headline)
    Only the components whose reference actually states a value. Several CGHD
    schematics were drawn with the Value field left blank -- C10's netlist is
    literally ``R1 R`` / ``C1 C`` -- and some sources carry an AC spec such as
    ``SINE(0 20m 400)`` that the photograph never shows. Scoring those measures
    the reference's completeness, not our reading.

With ``--gt`` (the default) the text strings come from CGHD's ``<text>``
annotations, so the result is the ceiling: how much value accuracy is available
if the handwriting is read perfectly. A real recogniser only has to close the
gap to it.

    python scripts/eval_values.py
    python scripts/eval_values.py --weights runs/yolov8s-2/weights/best.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import load_references, scope
from circuit2spice import ltspice
from circuit2spice.detect import detections_from_voc, nms_across_classes
from circuit2spice.netcompare import informative_values, value_f1
from circuit2spice.pipeline import _load_and_scale
from circuit2spice.topology import build_circuit
from circuit2spice.values import apply_to_graph, values_from_detections

ROOT = Path("data/cghd/drafter_1")
REF_NETS = Path("outputs/ref_netlists")


def reference_values() -> dict:
    out = {}
    for net in sorted(REF_NETS.glob("*.net")):
        out[net.stem] = ltspice.parse_netlist_values(net.read_text(errors="ignore"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="run the detector instead of ground-truth boxes")
    ap.add_argument("--out", default="outputs/values.json")
    a = ap.parse_args()

    detector = None
    if a.weights:
        from circuit2spice.detect import YoloDetector
        detector = YoloDetector(a.weights)

    refs = load_references()
    in_scope = scope(refs)
    refvals = reference_values()

    rows = []
    for xml in sorted((ROOT / "annotations").glob("*.xml")):
        circuit = xml.stem.split("_")[0]
        if not in_scope.get(circuit, True) or circuit not in refvals:
            continue
        img_p = ROOT / "images" / f"{xml.stem}.jpg"
        if not img_p.exists():
            continue

        dets = (detections_from_voc(xml) if detector is None
                else nms_across_classes(detector(str(img_p))))
        img, dets = _load_and_scale(img_p, dets, 1600)
        graph, _ = build_circuit(img, dets)

        defaults = {c.ref: c.value for c in graph.components}
        apply_to_graph(graph, values_from_detections(dets))
        read = {c.ref: c.value for c in graph.components}

        full = refvals[circuit]
        info = informative_values(full)
        rows.append({
            "stem": xml.stem, "circuit": circuit,
            "all_default": round(value_f1(defaults, full)[2], 4),
            "all_read": round(value_f1(read, full)[2], 4),
            "info_default": round(value_f1(defaults, info)[2], 4) if info else None,
            "info_read": round(value_f1(read, info)[2], 4) if info else None,
            "n_informative": len(info), "n_ref": len(full),
        })

    def mean(key, where=lambda r: True):
        vals = [r[key] for r in rows if r[key] is not None and where(r)]
        return round(statistics.mean(vals), 4) if vals else None

    per = collections.defaultdict(list)
    for r in rows:
        per[r["circuit"]].append(r)

    tag = "full-system" if detector else "gt-boxes"
    result = {
        tag: {
            "images": len(rows),
            "all": {"default": mean("all_default"), "read": mean("all_read")},
            "informative": {"default": mean("info_default"),
                            "read": mean("info_read"),
                            "images": sum(1 for r in rows if r["info_read"] is not None)},
            "per_circuit": {
                c: {"n": len(rs),
                    "default": round(statistics.mean(
                        [x["info_default"] for x in rs if x["info_default"] is not None] or [0]), 4),
                    "read": round(statistics.mean(
                        [x["info_read"] for x in rs if x["info_read"] is not None] or [0]), 4),
                    "scored": sum(1 for x in rs if x["info_read"] is not None)}
                for c, rs in sorted(per.items(), key=lambda t: int(t[0][1:]))
            },
        },
        "rows": rows,
    }

    print(f"{tag}: bill-of-values F1 over {len(rows)} in-scope images\n")
    print(f"{'circuit':8} {'n':>3} {'scored':>6} {'defaults':>9} {'read':>7}")
    for c, d in result[tag]["per_circuit"].items():
        print(f"{c:8} {d['n']:3d} {d['scored']:6d} {d['default']:9.3f} {d['read']:7.3f}")
    inf = result[tag]["informative"]
    print(f"\n  informative references ({inf['images']} images): "
          f"{inf['default']:.3f} -> {inf['read']:.3f}")
    print(f"  all references        ({len(rows)} images): "
          f"{result[tag]['all']['default']:.3f} -> {result[tag]['all']['read']:.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
