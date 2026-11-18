"""Convert CGHD PASCAL-VOC annotations into a YOLO dataset.

One decision here matters more than the rest: **the split is by drafter, not by
image.** CGHD has each drafter draw every circuit twice and photograph each
drawing four times, so a random image-level split puts near-duplicates of the
same drawing in both train and test. That inflates mAP dramatically while
measuring nothing useful. Holding out whole drafters measures what we actually
care about: does this work on a hand it has never seen?
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .classes import NAME_TO_ID, NAMES, merge_label

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_voc(xml_path: Path) -> tuple[list, tuple[int, int]]:
    """Return ``([(label, x1, y1, x2, y2), ...], (width, height))``."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        return [], (0, 0)
    w = int(float(size.findtext("width") or 0))
    h = int(float(size.findtext("height") or 0))
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin"))
            y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax"))
            y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((obj.findtext("name"), x1, y1, x2, y2))
    return boxes, (w, h)


def find_image(drafter_dir: Path, stem: str) -> Path | None:
    for suf in IMAGE_SUFFIXES:
        p = drafter_dir / "images" / f"{stem}{suf}"
        if p.is_file():
            return p
    return None


def drafter_dirs(root: Path) -> list[Path]:
    def key(p: Path) -> int:
        m = re.search(r"-?\d+", p.name)
        return int(m.group()) if m else 0

    return sorted((p for p in root.glob("drafter_*") if p.is_dir()), key=key)


def convert(
    root: str | Path,
    out: str | Path,
    *,
    n_test_drafters: int = 4,
    n_val_drafters: int = 3,
    link: bool = True,
) -> dict:
    """Build a YOLO dataset at ``out`` from the CGHD tree at ``root``."""
    root, out = Path(root), Path(out)
    drafters = drafter_dirs(root)
    if not drafters:
        raise FileNotFoundError(f"no drafter_* folders under {root}")

    test = set(drafters[-n_test_drafters:]) if n_test_drafters else set()
    val = set(drafters[-(n_test_drafters + n_val_drafters) : -n_test_drafters]) if n_val_drafters else set()
    # With too few drafters to hold any out, fall back to a single split so the
    # pipeline still runs end to end on a local subset.
    if len(drafters) <= n_test_drafters + n_val_drafters:
        test, val = set(), set()

    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = {"kept": 0, "dropped": 0, "images": {"train": 0, "val": 0, "test": 0},
             "per_class": {n: 0 for n in NAMES}, "dropped_labels": {}}

    for d in drafters:
        split = "test" if d in test else "val" if d in val else "train"
        for xml in sorted((d / "annotations").glob("*.xml")):
            img = find_image(d, xml.stem)
            if img is None:
                continue
            boxes, (W, H) = parse_voc(xml)
            if not W or not H:
                continue
            lines = []
            for raw, x1, y1, x2, y2 in boxes:
                merged = merge_label(raw)
                if merged is None:
                    stats["dropped"] += 1
                    stats["dropped_labels"][raw] = stats["dropped_labels"].get(raw, 0) + 1
                    continue
                x1, x2 = sorted((max(0.0, min(x1, W)), max(0.0, min(x2, W))))
                y1, y2 = sorted((max(0.0, min(y1, H)), max(0.0, min(y2, H))))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    stats["dropped"] += 1
                    continue
                lines.append(
                    f"{NAME_TO_ID[merged]} {(x1 + x2) / 2 / W:.6f} "
                    f"{(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                )
                stats["kept"] += 1
                stats["per_class"][merged] += 1
            if not lines:
                continue
            stem = f"{d.name}_{xml.stem}"
            dst = out / "images" / split / f"{stem}{img.suffix}"
            if not dst.exists():
                try:
                    if link:
                        dst.symlink_to(img.resolve())
                    else:
                        raise OSError
                except OSError:
                    import shutil

                    shutil.copy2(img, dst)
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            stats["images"][split] += 1

    yaml_text = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    )
    (out / "data.yaml").write_text(yaml_text)
    (out / "split.json").write_text(
        json.dumps(
            {"test": [d.name for d in test], "val": [d.name for d in val],
             "train": [d.name for d in drafters if d not in test and d not in val]},
            indent=2,
        )
    )
    return stats
