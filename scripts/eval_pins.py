"""Score transistor pin roles -- the thing exact match cannot see.

``netlist_to_graph`` is undirected and ``_signatures`` sorts each component's
legs, so a BJT with its collector and emitter transposed is isomorphic to the
correct circuit. That is the right answer to "is the wiring the same" and the
wrong answer to "will it simulate": an NPN wired in reverse-active has a beta
of a few rather than a hundred, so the bias point is wrong while every headline
number stays green.

This measures it directly. Net names do not correspond between our netlist and
LTspice's, so the correspondence comes from the isomorphism itself, which also
means only images whose topology already matches can be scored -- without a
mapping between nets there is nothing to compare.

    python scripts/eval_pins.py
    python scripts/eval_pins.py --positional     # the old behaviour, for the delta
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
from circuit2spice.netcompare import compare, pin_role_accuracy
from circuit2spice.pipeline import _load_and_scale
from circuit2spice.topology import build_circuit

ROOT = Path("data/cghd/drafter_1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--method", default="unet")
    ap.add_argument("--positional", action="store_true",
                    help="disable arrowhead reading, to measure what it bought")
    ap.add_argument("--out", default="outputs/pins.json")
    a = ap.parse_args()

    if a.positional:
        import circuit2spice.topology as topo
        topo.order_transistor_terminals = lambda binary, det, terms: terms

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
        if not any(r[0].upper() in "QM" for r in refs[circuit]):
            continue                       # nothing with more than two pins
        dets = (detections_from_voc(xml) if detector is None
                else nms_across_classes(detector(str(img_p))))
        img, dets = _load_and_scale(img_p, dets, 1600)
        g, _ = build_circuit(img, dets, bridge_px=6, method=a.method)
        pred = predicted_elements(g)
        has_gnd = any(d.cls == "gnd" for d in dets)
        ok, tot = pin_role_accuracy(pred, refs[circuit], pin_ground=has_gnd)
        rows.append({"stem": xml.stem, "circuit": circuit, "correct": ok,
                     "scored": tot,
                     "exact": compare(xml.stem, circuit, pred, refs[circuit],
                                      pin_ground=has_gnd).exact})
        per[circuit][0] += ok
        per[circuit][1] += tot

    ok = sum(r["correct"] for r in rows)
    tot = sum(r["scored"] for r in rows)
    tag = ("positional" if a.positional else "arrowhead")
    tag += "/full-system" if detector else "/gt-boxes"
    result = {
        tag: {
            "images_with_multipin_parts": len(rows),
            "pins_scored": tot, "pins_correct": ok,
            "accuracy": round(ok / tot, 4) if tot else None,
            "per_circuit": {c: {"correct": v[0], "scored": v[1]}
                            for c, v in sorted(per.items(), key=lambda t: int(t[0][1:]))},
            "note": ("only images whose topology already matches can be scored; "
                     "without an isomorphism there is no net correspondence"),
        },
        "rows": rows,
    }
    print(f"{tag}: {len(rows)} in-scope images contain a multi-pin part")
    for c, v in result[tag]["per_circuit"].items():
        print(f"  {c:5} {v['correct']}/{v['scored']}")
    print(f"  pin-role accuracy: {ok}/{tot}" + (f" = {ok/tot:.3f}" if tot else ""))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(result)
    out.write_text(json.dumps(merged, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
