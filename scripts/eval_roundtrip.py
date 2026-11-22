"""Does the schematic we emit actually survive LTspice? Measured, not assumed.

Every other score in this project compares our netlist to a reference. This
one asks the question a user asks: open the ``.asc`` we wrote, have LTspice
extract its netlist, and check LTspice sees the same circuit we meant. It is
the claim the README makes -- "an LTspice .asc that opens and simulates" --
and the one that is cheapest to believe without checking, because a .asc
whose wires do not meet opens perfectly happily with the component floating.

Scored over the in-scope circuits only, like everything else here.

    python scripts/eval_roundtrip.py
    python scripts/eval_roundtrip.py --weights runs/yolov8s-2/weights/best.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import load_references, scope
from circuit2spice import ltspice
from circuit2spice.pipeline import process

ROOT = Path("data/cghd/drafter_1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--method", default="unet")
    ap.add_argument("--analysis", default=".op")
    ap.add_argument("--out", default="outputs/roundtrip.json")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if not ltspice.available():
        print("LTspice is not installed; nothing to measure")
        return 1

    detector = None
    if a.weights:
        from circuit2spice.detect import YoloDetector
        detector = YoloDetector(a.weights)

    in_scope = scope(load_references())
    xmls = [x for x in sorted(ROOT.glob("annotations/*.xml"))
            if in_scope.get(x.stem.split("_")[0], False)]
    if a.limit:
        xmls = xmls[:a.limit]

    out_dir = Path("outputs/roundtrip")
    rows, per = [], collections.defaultdict(lambda: [0, 0])
    t0 = time.time()
    for i, xml in enumerate(xmls):
        circuit = xml.stem.split("_")[0]
        img = ROOT / "images" / f"{xml.stem}.jpg"
        if not img.exists():
            continue
        try:
            res = process(img, detector=detector,
                          gt_xml=None if detector else xml,
                          out_dir=out_dir, stem=xml.stem, method=a.method,
                          analysis=a.analysis, read_values=False, verify=True)
            v = res.verification
            ok = bool(v is not None and v.ok)
            note = v.summary() if v is not None else "no verification"
        except Exception as exc:                      # noqa: BLE001
            ok, note = False, f"{type(exc).__name__}: {exc}"
        rows.append({"stem": xml.stem, "circuit": circuit, "ok": ok,
                     "note": note})
        per[circuit][1] += 1
        per[circuit][0] += int(ok)
        if i % 8 == 0:
            print(f"  {i}/{len(xmls)}  ({time.time() - t0:.0f}s)", flush=True)

    ok = sum(r["ok"] for r in rows)
    tag = "full-system" if detector else "gt-boxes"
    print(f"\n{tag}: LTspice round-trip over {len(rows)} in-scope images")
    for c in sorted(per, key=lambda x: int(x[1:])):
        print(f"  {c:4} {per[c][0]}/{per[c][1]}")
    print(f"  round-trip OK: {ok}/{len(rows)}"
          + (f" = {ok / len(rows):.3f}" if rows else ""))

    result = {tag: {"images": len(rows), "ok": ok,
                    "rate": round(ok / len(rows), 4) if rows else None,
                    "per_circuit": {c: {"ok": v[0], "n": v[1]}
                                    for c, v in sorted(per.items(),
                                                       key=lambda t: int(t[0][1:]))},
                    "analysis": a.analysis, "method": a.method},
              "rows": rows}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(p.read_text()) if p.exists() else {}
    merged.update(result)
    p.write_text(json.dumps(merged, indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
