"""Evaluate on drafters the detector has never seen.

The end-to-end benchmark is stuck at drafter 1, because CGHD ships reference
netlists for that drafter and no other. That leaves one open question the
headline number cannot answer: *does any of this work on somebody else's
handwriting?*

Two things can be measured on held-out drafters without reference netlists:

**True test mAP.** These drafters were the detector's test split, so their
bounding boxes give the honest detection score. The 0.807 reported elsewhere is
*validation* mAP, which is the weaker claim.

**Structural quality of the extracted circuit.** No reference netlist is needed
to notice that a circuit came out in three disconnected pieces. Validated
against ground truth on drafter 1, connectivity correlates +0.63 with actually
recovering the right circuit (0.976 when correct, 0.705 when wrong), so it is a
real signal -- but it is a *proxy*, not proof, and is reported as such.

    python scripts/eval_heldout.py --drafters 29 30 31 32 \
        --weights runs/yolov8s-2/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit2spice.classes import NAME_TO_ID, NAMES, merge_label
from circuit2spice.detect import detections_from_voc, nms_across_classes
from circuit2spice.evaluate import TopologyMetrics, score_graph
from circuit2spice.pipeline import _load_and_scale
from circuit2spice.topology import build_circuit
from circuit2spice.voc2yolo import find_image, parse_voc

ROOT = Path("data/cghd")


def build_yolo_split(drafters: list, out: Path, name_to_id: dict | None = None) -> Path:
    """Lay the given drafters out as a YOLO 'val' split, for mAP.

    ``name_to_id`` must come from the *model being evaluated*, not from
    ``classes.py``. The trained weights carry 21 classes rather than 16 -- a
    Colab edit appended to the name list instead of replacing it, leaving
    duplicates so that junction/crossover/switch/terminal/text live at 15-19
    while our local taxonomy puts them at 10-14. Writing ground truth with the
    local ids silently scores the five commonest classes against the wrong
    index, and junction alone is a third of all boxes.
    """
    import shutil

    name_to_id = name_to_id or NAME_TO_ID

    for s in ("images", "labels"):
        d = out / s / "val"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    n = 0
    for name in drafters:
        d = ROOT / name
        for xml in sorted((d / "annotations").glob("*.xml")):
            img = find_image(d, xml.stem)
            if img is None:
                continue
            boxes, (W, H) = parse_voc(xml)
            if not W or not H:
                continue
            lines = []
            for raw, x1, y1, x2, y2 in boxes:
                m = merge_label(raw)
                if m is None:
                    continue
                x1, x2 = sorted((max(0.0, min(x1, W)), max(0.0, min(x2, W))))
                y1, y2 = sorted((max(0.0, min(y1, H)), max(0.0, min(y2, H))))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                if m not in name_to_id:
                    continue
                lines.append(
                    f"{name_to_id[m]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                    f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            if not lines:
                continue
            stem = f"{name}_{xml.stem}"
            shutil.copy2(img, out / "images" / "val" / f"{stem}{img.suffix}")
            (out / "labels" / "val" / f"{stem}.txt").write_text("\n".join(lines))
            n += 1

    yaml = (f"path: {out.resolve().as_posix()}\n"
            "train: images/val\nval: images/val\n"
            "names:\n" + "".join(f"  {i}: {nm}\n" for i, nm in enumerate(NAMES)))
    (out / "data.yaml").write_text(yaml)
    print(f"  staged {n} images from {len(drafters)} held-out drafters")
    return out / "data.yaml"


def structural(drafters: list, detector=None, limit_per: int | None = None) -> dict:
    """Per-image connectivity of the extracted circuit."""
    vals, comps, imgs = [], 0, 0
    for name in drafters:
        d = ROOT / name
        xmls = sorted((d / "annotations").glob("*.xml"))
        if limit_per:
            xmls = xmls[:limit_per]
        for xml in xmls:
            img_p = find_image(d, xml.stem)
            if img_p is None:
                continue
            dets = (detections_from_voc(xml) if detector is None
                    else nms_across_classes(detector(str(img_p))))
            im, dets = _load_and_scale(img_p, dets, 1600)
            g, _ = build_circuit(im, dets)
            if not g.components:
                continue
            m = score_graph(g, TopologyMetrics())
            vals.append(m.connected_frac)
            comps += len(g.components)
            imgs += 1
    return {
        "images": imgs,
        "components": comps,
        "connected_frac_mean": round(statistics.mean(vals), 4) if vals else 0.0,
        "fully_connected_rate": round(sum(v >= 0.999 for v in vals) / len(vals), 4) if vals else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafters", nargs="+", default=["29", "30", "31", "32"])
    ap.add_argument("--weights", default="runs/yolov8s-2/weights/best.pt")
    ap.add_argument("--limit-per", type=int, default=None)
    ap.add_argument("--skip-map", action="store_true")
    ap.add_argument("--out", default="outputs/heldout.json")
    a = ap.parse_args()

    names = [f"drafter_{d}" for d in a.drafters]
    missing = [n for n in names if not (ROOT / n / "annotations").is_dir()]
    if missing:
        print(f"missing locally: {missing}\n  run: python scripts/fetch_subset.py "
              + " ".join(a.drafters))
        return 1

    res = {"drafters": names}

    if not a.skip_map and Path(a.weights).is_file():
        print("true test-set mAP (held-out drafters):")
        from ultralytics import YOLO
        model = YOLO(a.weights)
        # Use the model's OWN class mapping, not the local taxonomy.
        m2i = {n: i for i, n in model.names.items()}
        print(f"  model has {len(model.names)} classes; "
              f"junction -> id {m2i.get('junction')}")
        data = build_yolo_split(names, Path("data/yolo_heldout"), m2i)
        r = model.val(data=str(data), split="val", verbose=False)
        res["map50_test"] = round(float(r.box.map50), 4)
        res["map50_95_test"] = round(float(r.box.map), 4)
        print(f"  mAP@50 = {res['map50_test']}   mAP@50-95 = {res['map50_95_test']}")

    print("\nstructural quality on unseen handwriting:")
    res["gt_boxes"] = structural(names, None, a.limit_per)
    print(f"  ground-truth boxes : {res['gt_boxes']}")
    if Path(a.weights).is_file():
        from circuit2spice.detect import YoloDetector
        res["full_system"] = structural(names, YoloDetector(a.weights), a.limit_per)
        print(f"  full system        : {res['full_system']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
    print("NOTE: connectivity is a proxy (r=+0.63 vs exact match on drafter 1), "
          "not a correctness guarantee.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
