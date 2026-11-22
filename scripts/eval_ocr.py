# -*- coding: utf-8 -*-
"""Measure a handwriting backend on a drawing whose labels we know.

A reading counts as correct only when **both** halves of the label parse to
the truth -- the designator and the value -- because a netlist is wrong if
either is. That is a stricter test than string equality in one direction
(``R210r`` passes, since it parses to R2 and 10 ohms) and stricter in the
other (a reading that parses to the wrong index fails even if it looks close).

    python scripts/eval_ocr.py --backend ollama
    python scripts/eval_ocr.py --backend vlm --model Qwen/Qwen2-VL-2B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_WEIGHTS = "runs/yolov8s-2/weights/best.pt"
#: How far a read box may sit from a truth point and still be that label.
MATCH_PX = 20


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="data/own/testing2.jpeg")
    ap.add_argument("--truth", default="data/own/testing2.labels.json")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "ollama", "vlm", "scored"])
    ap.add_argument("--model", default=None,
                    help="checkpoint or Ollama tag, overriding the default")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default="outputs/ocr.json")
    a = ap.parse_args(argv)

    from circuit2spice.detect import YoloDetector, nms_across_classes
    from circuit2spice.ocr import read_structured
    from circuit2spice.pipeline import _load_and_scale
    from circuit2spice.values import parse_label

    truth = json.loads(Path(a.truth).read_text(encoding="utf-8"))
    want = {(l["x"], l["y"]): l["text"] for l in truth["labels"]}

    dets = nms_across_classes(YoloDetector(a.weights)(a.image))
    img, dets = _load_and_scale(a.image, dets, 1600)
    boxes = [d for d in dets if d.cls == "text"]

    t0 = time.perf_counter()
    n = read_structured(img, dets, model_name=a.model, backend=a.backend)
    dt = time.perf_counter() - t0

    rows, good, bad = [], 0, 0
    for d in sorted(boxes, key=lambda x: (x.cy, x.cx)):
        key = min(want, key=lambda k: (k[0] - d.x1) ** 2 + (k[1] - d.y1) ** 2)
        near = (key[0] - d.x1) ** 2 + (key[1] - d.y1) ** 2 <= MATCH_PX ** 2
        truth_text = want[key] if near else None
        got = getattr(d, "text", "") or ""
        wd, wv = parse_label(truth_text) if truth_text else (None, None)
        pd, pv = parse_label(got)
        ok = bool(truth_text) and pd == wd and \
            (pv.spice if pv else None) == (wv.spice if wv else None)
        good += ok
        bad += bool(truth_text) and not ok
        rows.append({"truth": truth_text, "read": got,
                     "designator": pd, "value": pv.spice if pv else None,
                     "correct": ok})
        mark = "OK " if ok else ("   " if truth_text is None else "XX ")
        print(f"  {mark} want {str(truth_text):10} got {got!r:20} "
              f"-> {pd}/{pv.spice if pv else None}")

    total = len(want)
    res = {"image": a.image, "backend": a.backend, "model": a.model,
           "labels": total, "read": n, "boxes": len(boxes),
           "correct": good, "wrong": bad,
           "accuracy": good / total if total else 0.0,
           "seconds": round(dt, 1),
           "seconds_per_label": round(dt / max(n, 1), 1), "rows": rows}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{good}/{total} correct ({res['accuracy']:.3f}), {bad} wrong, "
          f"{res['seconds']}s = {res['seconds_per_label']}s per label")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
