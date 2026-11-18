"""HOG + SVM component classifier -- the baseline from the reference work.

Aksakalli's ``circuit_recognizer`` classifies resistor/diode/inductor with an
SVM over HOG features; Liu & Xiao use Fourier descriptors into an SVM; Roy et
al. use texture and shape features into kNN/Naive Bayes. All of them report
high accuracy (93-96%) -- but all of them report it on *pre-segmented, isolated
component crops*, which is a much easier problem than finding the components in
a photograph of a whole circuit.

Reimplementing that baseline here lets the report compare like with like on the
same data: this classifier gets ground-truth crops handed to it, exactly as in
the papers, while the detector has to find them itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import PLACEABLE
from .voc2yolo import drafter_dirs, find_image, parse_voc
from .classes import merge_label

CROP = 64


def _hog(img: np.ndarray) -> np.ndarray:
    from skimage.feature import hog

    return hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_crops(root: str | Path, classes: set | None = None) -> tuple:
    """Return ``(X_hog, y, groups)`` over ground-truth component crops."""
    import cv2

    root = Path(root)
    classes = classes or set(PLACEABLE)
    X, y, groups = [], [], []
    for d in drafter_dirs(root):
        for xml in sorted((d / "annotations").glob("*.xml")):
            img_p = find_image(d, xml.stem)
            if img_p is None:
                continue
            boxes, _ = parse_voc(xml)
            if not boxes:
                continue
            img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            for raw, x1, y1, x2, y2 in boxes:
                m = merge_label(raw)
                if m not in classes:
                    continue
                xa, ya = max(0, int(x1)), max(0, int(y1))
                xb, yb = min(w, int(x2)), min(h, int(y2))
                if xb - xa < 8 or yb - ya < 8:
                    continue
                crop = cv2.resize(img[ya:yb, xa:xb], (CROP, CROP))
                crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
                X.append(_hog(crop))
                y.append(m)
                # Group by circuit so train/test never share a drawing.
                groups.append(f"{d.name}/{xml.stem.split('_')[0]}")
    return np.asarray(X), np.asarray(y), np.asarray(groups)


@dataclass
class BaselineResult:
    accuracy: float
    macro_f1: float
    n_train: int
    n_test: int
    labels: list
    confusion: list
    report: dict


def run(root: str | Path, test_frac: float = 0.3, seed: int = 0) -> BaselineResult:
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, accuracy_score)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    X, y, groups = extract_crops(root)
    if len(X) == 0:
        raise RuntimeError(f"no component crops found under {root}")

    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_groups = set(uniq[:n_test])
    mask = np.array([g in test_groups for g in groups])

    clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000))
    clf.fit(X[~mask], y[~mask])
    pred = clf.predict(X[mask])

    labels = sorted(set(y.tolist()))
    return BaselineResult(
        accuracy=float(accuracy_score(y[mask], pred)),
        macro_f1=float(f1_score(y[mask], pred, average="macro", zero_division=0)),
        n_train=int((~mask).sum()),
        n_test=int(mask.sum()),
        labels=labels,
        confusion=confusion_matrix(y[mask], pred, labels=labels).tolist(),
        report=classification_report(y[mask], pred, labels=labels,
                                     zero_division=0, output_dict=True),
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cghd")
    ap.add_argument("--out", default="outputs/baseline_svm.json")
    a = ap.parse_args()
    r = run(a.root)
    print(f"HOG+SVM on ground-truth crops: accuracy={r.accuracy:.4f} "
          f"macro-F1={r.macro_f1:.4f}  (train={r.n_train}, test={r.n_test})")
    for lab in r.labels:
        d = r.report.get(lab, {})
        print(f"  {lab:16} P={d.get('precision',0):.3f} R={d.get('recall',0):.3f} "
              f"F1={d.get('f1-score',0):.3f}  n={int(d.get('support',0))}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r.__dict__, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
