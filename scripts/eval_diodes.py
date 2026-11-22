"""Score diode polarity end to end -- the other thing exact match cannot see.

``scripts/diode_orient.py --cv`` measures the model on crops. This measures
the pipeline: photograph in, netlist out, is each diode's ``(anode, cathode)``
the reference's. The two are different questions, because between them sit
terminal finding and the agreement gate, either of which can decline.

As with ``eval_pins.py``, net names do not correspond between our netlist and
LTspice's, so the correspondence comes from the isomorphism itself -- only
images whose topology already matches can be scored at all.

Drafter 1 ships the only reference schematics, and of its in-scope circuits
C8 and C9 are the ones containing diodes, so that is the whole scorable set.
It is small. It is also the only end-to-end diode evidence that exists.

    python scripts/eval_diodes.py
    python scripts/eval_diodes.py --blind      # model off, to measure what it bought
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import load_references, predicted_elements, scope
from circuit2spice.detect import detections_from_voc, nms_across_classes
from circuit2spice.netcompare import compare, polarity_accuracy
from circuit2spice.pipeline import _load_and_scale
from circuit2spice.topology import build_circuit

ROOT = Path("data/cghd/drafter_1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--method", default="unet")
    ap.add_argument("--blind", action="store_true",
                    help="leave the terminal order alone, to measure the delta")
    ap.add_argument("--out", default="outputs/diodes.json")
    a = ap.parse_args()

    if a.blind:
        import circuit2spice.topology as topo
        topo.order_diode_terminals = lambda bgr, det, terms: terms

    detector = None
    if a.weights:
        from circuit2spice.detect import YoloDetector
        detector = YoloDetector(a.weights)

    refs = load_references()
    in_scope = scope(refs)
    rows, per = [], collections.defaultdict(lambda: [0, 0])

    for xml in sorted(ROOT.glob("annotations/*.xml")):
        circuit = xml.stem.split("_")[0]
        img_p = ROOT / "images" / f"{xml.stem}.jpg"
        if circuit not in refs or not in_scope.get(circuit, True) or not img_p.exists():
            continue
        if not any(r[:1].upper() == "D" for r in refs[circuit]):
            continue                       # no diode to score
        dets = (detections_from_voc(xml) if detector is None
                else nms_across_classes(detector(str(img_p))))
        img, dets = _load_and_scale(img_p, dets, 1600)
        g, _ = build_circuit(img, dets, bridge_px=6, method=a.method)
        pred = predicted_elements(g)
        has_gnd = any(d.cls == "gnd" for d in dets)
        ok, tot = polarity_accuracy(pred, refs[circuit], pin_ground=has_gnd)
        rows.append({"stem": xml.stem, "circuit": circuit, "correct": ok,
                     "scored": tot,
                     "exact": compare(xml.stem, circuit, pred, refs[circuit],
                                      pin_ground=has_gnd).exact})
        per[circuit][0] += ok
        per[circuit][1] += tot

    ok = sum(r["correct"] for r in rows)
    tot = sum(r["scored"] for r in rows)
    tag = ("blind" if a.blind else "learned")
    tag += "/full-system" if detector else "/gt-boxes"
    result = {
        tag: {
            "images_with_diodes": len(rows),
            "diodes_scored": tot, "diodes_correct": ok,
            "accuracy": round(ok / tot, 4) if tot else None,
            "per_circuit": {c: {"correct": v[0], "scored": v[1]}
                            for c, v in sorted(per.items(),
                                               key=lambda t: int(t[0][1:]))},
            "note": ("only images whose topology already matches can be scored; "
                     "without an isomorphism there is no net correspondence"),
        },
        "rows": rows,
    }
    print(f"{tag}: {len(rows)} in-scope images contain a diode")
    for c, v in result[tag]["per_circuit"].items():
        print(f"  {c:5} {v['correct']}/{v['scored']}")
    print(f"  diode polarity: {ok}/{tot}" + (f" = {ok/tot:.3f}" if tot else ""))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(result)
    out.write_text(json.dumps(merged, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
