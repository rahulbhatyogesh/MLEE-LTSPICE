"""The learned stroke segmenter, and the threshold that makes it work."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from circuit2spice import segment
from circuit2spice.wires import binarize, build_node_map


def test_default_threshold_is_not_a_half():
    """Regression guard with a reason, because 0.5 looks like the obvious value.

    CGHD's masks disagree about printed page ruling -- some annotators erased
    it, some kept it -- so the network is merely *uncertain* over a rule line
    rather than confident it is paper. At 0.5 that uncertainty reads as ink and
    the ruling comes straight back, which measured 0.500 exact match against
    0.778 at the tuned value.
    """
    assert segment.DEFAULT_THRESHOLD > 0.9


def test_threshold_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("CIRCUIT2SPICE_UNET_THR", "0.5")
    reloaded = importlib.reload(segment)
    try:
        assert reloaded.DEFAULT_THRESHOLD == 0.5
    finally:
        monkeypatch.delenv("CIRCUIT2SPICE_UNET_THR")
        importlib.reload(segment)


def _spy(monkeypatch):
    seen = {}

    def fake(bgr, model_path=None, *, long_side=1024, threshold=None):
        seen["threshold"] = threshold
        return np.zeros(bgr.shape[:2], np.uint8)

    monkeypatch.setattr(segment, "segment_strokes", fake)
    return seen


def test_binarize_forwards_the_threshold(monkeypatch):
    seen = _spy(monkeypatch)
    binarize(np.zeros((64, 64, 3), np.uint8), method="unet", unet_thr=0.99)
    assert seen["threshold"] == 0.99


def test_binarize_leaves_the_default_alone_when_unset(monkeypatch):
    """None must reach the segmenter, not a hard-coded 0.5 copied en route."""
    seen = _spy(monkeypatch)
    binarize(np.zeros((64, 64, 3), np.uint8), method="unet")
    assert seen["threshold"] is None


def test_build_node_map_forwards_the_threshold(monkeypatch):
    seen = _spy(monkeypatch)
    build_node_map(np.zeros((64, 64, 3), np.uint8), [], method="unet", unet_thr=0.97)
    assert seen["threshold"] == 0.97


@pytest.mark.skipif(not segment.available(), reason="no exported U-Net available")
def test_raising_the_threshold_removes_page_ruling():
    """On ruled paper the rules are the first thing to go as the cut-off rises.

    A printed rule is inked across nearly the whole page, which no part of a
    circuit is, so counting page-wide rows isolates the ruling from the drawing.
    """
    import cv2

    page = np.full((400, 600, 3), 245, np.uint8)
    for y in range(40, 400, 60):                       # the ruling
        cv2.line(page, (0, y), (599, y), (205, 205, 215), 1)
    cv2.rectangle(page, (150, 120), (450, 280), (40, 40, 40), 3)   # the drawing

    def page_wide_rows(mask):
        w = mask.shape[1]
        n = 0
        for row in mask > 0:
            idx = np.flatnonzero(row)
            if idx.size and idx[-1] - idx[0] >= 0.7 * w and idx.size >= 0.35 * w:
                n += 1
        return n

    low = segment.segment_strokes(page, threshold=0.5)
    high = segment.segment_strokes(page, threshold=segment.DEFAULT_THRESHOLD)
    assert page_wide_rows(high) <= page_wide_rows(low)
    assert high.shape == page.shape[:2]


def test_the_low_threshold_path_is_off_by_default():
    """It was measured and it loses; it stays available, not enabled."""
    assert segment.DEFAULT_LOW_THRESHOLD is None


def test_the_low_threshold_only_ever_adds_ink(monkeypatch):
    """Union with a thickness-filtered mask: it cannot remove a confident pixel."""
    import cv2

    page = np.full((256, 256, 3), 240, np.uint8)
    cv2.rectangle(page, (60, 60), (200, 200), (30, 30, 30), 4)
    for y in range(20, 256, 40):
        cv2.line(page, (0, y), (255, y), (200, 200, 205), 1)

    plain = segment.segment_strokes(page)
    hybrid = segment.segment_strokes(page, low_threshold=0.5, thin_q=70.0)
    assert ((plain > 0) & ~(hybrid > 0)).sum() == 0
