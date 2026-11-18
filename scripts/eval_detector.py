"""Fold trained YOLO weights into the results: detector mAP + full-system benchmark.

Run once ``best.pt`` is back from Colab:

    python scripts/eval_detector.py --weights runs/yolov8s/weights/best.pt

Writes ``outputs/detector.json``, which the report and poster builders read.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=None,
                    help="data.yaml for mAP; omit to skip the detection metrics")
    ap.add_argument("--out", default="outputs/detector.json")
    a = ap.parse_args()

    out = {"weights": str(a.weights)}

    if a.data:
        from ultralytics import YOLO
        m = YOLO(a.weights)
        r = m.val(data=a.data, split="test", verbose=False)
        out["map50"] = float(r.box.map50)
        out["map50_95"] = float(r.box.map)
        names = getattr(r, "names", {}) or {}
        out["per_class_ap50"] = {
            str(names.get(i, i)): float(v) for i, v in enumerate(r.box.ap50)
        }
        print(f"detector: mAP@50={out['map50']:.4f}  mAP@50:95={out['map50_95']:.4f}")

    # Full system: detector + wire tracing, scored against reference netlists.
    from benchmark import run, summarise
    from circuit2spice.detect import YoloDetector

    det = YoloDetector(a.weights)
    rows = run(None, det, None, "flatfield", "none", 50.0)
    s = summarise(rows, "full system (detector + topology)")
    out.update({"exact": s["exact"], "bom_f1": s["bom_f1"],
                "sig_f1": s["sig_f1"], "images": s["images"]})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")
    print("now rerun:  python scripts/build_report.py && python scripts/build_poster.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
