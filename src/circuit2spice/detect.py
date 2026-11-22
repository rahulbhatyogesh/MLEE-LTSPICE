"""Symbol detection.

Two sources of detections are supported and they are deliberately
interchangeable:

``YoloDetector``
    The trained YOLOv8 model.

``detections_from_voc``
    CGHD's ground-truth boxes, used as a *perfect detector*.

Being able to swap in ground truth matters for more than convenience. It
separates "the detector missed a capacitor" from "the wire tracing joined two
nodes that should be separate", which is the difference between a reportable
result and a guess -- and it let the whole downstream pipeline be built and
tested before the model finished training.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .classes import NAMES, merge_label
from .graph import Detection
from .voc2yolo import parse_voc, parse_voc_text


def detections_from_voc(xml_path: str | Path, min_conf: float = 1.0) -> list:
    """Ground-truth detections from a CGHD annotation file."""
    boxes, _size = parse_voc(Path(xml_path))
    strings = parse_voc_text(Path(xml_path))
    out = []
    for raw, x1, y1, x2, y2 in boxes:
        merged = merge_label(raw)
        if merged is None:
            continue
        out.append(Detection(merged, x1, y1, x2, y2, min_conf,
                             text=strings.get((x1, y1, x2, y2), "")))
    return out


class YoloDetector:
    """Thin wrapper around a trained Ultralytics model."""

    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 896):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        # Prefer the names baked into the weights; fall back to our taxonomy.
        self.names = getattr(self.model, "names", None) or {
            i: n for i, n in enumerate(NAMES)
        }

    def __call__(self, image, conf: float | None = None) -> list:
        res = self.model.predict(
            source=image,
            conf=self.conf if conf is None else conf,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        out = []
        if res.boxes is None:
            return out
        for xyxy, cls_id, c in zip(
            res.boxes.xyxy.cpu().numpy(),
            res.boxes.cls.cpu().numpy().astype(int),
            res.boxes.conf.cpu().numpy(),
        ):
            name = self.names.get(int(cls_id), str(cls_id))
            out.append(
                Detection(name, float(xyxy[0]), float(xyxy[1]),
                          float(xyxy[2]), float(xyxy[3]), float(c))
            )
        return out


def nms_across_classes(dets: list, iou_thr: float = 0.75) -> list:
    """Drop near-duplicate boxes that differ only in class."""
    dets = sorted(dets, key=lambda d: -d.conf)
    keep: list = []
    for d in dets:
        if any(_iou(d, k) > iou_thr for k in keep):
            continue
        keep.append(d)
    return keep


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def scale_detections(dets: list, sx: float, sy: float) -> list:
    # Carry every field across: rebuilding positionally silently dropped the
    # transcribed text, so value reading worked on images small enough to skip
    # rescaling and quietly did nothing on the rest.
    return [
        replace(d, x1=d.x1 * sx, y1=d.y1 * sy, x2=d.x2 * sx, y2=d.y2 * sy)
        for d in dets
    ]


def draw(image: np.ndarray, dets: list) -> np.ndarray:
    """Overlay detections, for qualitative figures."""
    import cv2

    palette = {}
    rng = np.random.default_rng(7)
    out = image.copy()
    thick = max(1, int(round(min(out.shape[:2]) / 400)))
    for d in dets:
        col = palette.setdefault(
            d.cls, tuple(int(v) for v in rng.integers(60, 255, 3))
        )
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, thick)
        cv2.putText(out, d.cls, (int(d.x1), max(12, int(d.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, col, thick)
    return out
