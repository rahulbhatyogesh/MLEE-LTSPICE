"""The diode orientation model: its label convention, its algebra, its gate.

The model itself is measured by ``scripts/diode_orient.py --cv``; what is
tested here is everything around it that a silent mistake could ruin. The
first test is the one that matters, because getting it wrong is exactly how
the first attempt scored at chance: it pins the training label to CGHD's own
definition of ``<rotation>`` rather than to anybody's reading of it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from circuit2spice.diode_net import (
    MIN_CONFIDENCE, SIZE, augment, crop_symbol, order_diode_terminals,
    rotate_crop,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diode_orient import anode_direction  # noqa: E402


# --------------------------------------------------------------- the label

def _cghd_anode_bearing(rotation: float) -> float:
    """Where CGHD's own code puts the anode, reimplemented from the dataset.

    This is ``utils.py::transform`` applied to the ``anode`` port that
    ``classes_ports.json`` places at unit ``(0, 0.5)``: centre it, flip y,
    rotate by ``-rotation``. The box is taken square so the anisotropic scale
    cannot rotate the result.
    """
    px, py = 0.0 - 0.5, 0.5 - 0.5          # shift to [-0.5, 0.5]^2
    px, py = px * 1.0, py * -1.0           # flip
    a = math.radians(-rotation)            # rotate(p, -radians(rotation))
    rx = px * math.cos(a) - py * math.sin(a)
    ry = px * math.sin(a) + py * math.cos(a)
    return math.degrees(math.atan2(ry, rx)) % 360.0


@pytest.mark.parametrize("rot", [0, 37, 45, 90, 103, 180, 225, 270, 343, 359])
def test_label_matches_cghd_own_transform(rot):
    """The training label is the dataset's definition, not a guess at it.

    Worth stating plainly: at 90 and 270 degrees ``180 - r`` equals ``r``, so
    a vertical diode cannot tell the correct convention from the naive one.
    Drafter 1's diodes are nearly all vertical, which is why eyeballing them
    "confirmed" the wrong rule and the result came out at chance everywhere
    else. The horizontal cases in this list are the ones with teeth.
    """
    want = _cghd_anode_bearing(rot)
    got = anode_direction(rot)
    assert abs((got - want + 180) % 360 - 180) < 1e-6, (rot, got, want)


def test_naive_convention_is_wrong_except_on_vertical_diodes():
    """Pin the trap itself, so it cannot be quietly reintroduced."""
    for rot in (90, 270):
        assert abs(anode_direction(rot) - rot) < 1e-6
    for rot in (0, 45, 180, 343):
        assert abs((anode_direction(rot) - rot + 180) % 360 - 180) > 1.0


# ------------------------------------------------------------- the algebra

def _bearing_of_blob(img):
    ys, xs = np.nonzero(img > 128)
    c = SIZE / 2 - 0.5
    return math.degrees(math.atan2(ys.mean() - c, xs.mean() - c)) % 360.0


def _blob_at(theta, radius=30):
    import cv2

    img = np.zeros((SIZE, SIZE), np.uint8)
    c = SIZE / 2 - 0.5
    cv2.circle(img, (int(round(c + radius * math.cos(math.radians(theta)))),
                     int(round(c + radius * math.sin(math.radians(theta))))),
               5, 255, -1)
    return img


@pytest.mark.parametrize("theta", [0, 37, 90, 150, 210, 300])
@pytest.mark.parametrize("deg", [0, 45, 90, 180, 270])
def test_rotating_the_crop_turns_the_label_the_other_way(theta, deg):
    """``augment`` leans on ``theta - a``; a sign slip here is unfindable later."""
    got = _bearing_of_blob(rotate_crop(_blob_at(theta), deg))
    assert abs((got - (theta - deg) + 180) % 360 - 180) < 4.0


@pytest.mark.parametrize("theta", [0, 37, 90, 150, 210, 300])
def test_mirrors_act_on_the_label_in_closed_form(theta):
    img = _blob_at(theta)
    lr = _bearing_of_blob(img[:, ::-1].copy())
    assert abs((lr - (180.0 - theta) + 180) % 360 - 180) < 4.0
    ud = _bearing_of_blob(img[::-1, :].copy())
    assert abs((ud - (-theta) + 180) % 360 - 180) < 4.0


def test_augment_returns_a_usable_crop_and_a_wrapped_angle():
    rng = np.random.default_rng(0)
    img = _blob_at(42)
    for _ in range(25):
        out, t = augment(img, 42.0, rng)
        assert out.shape == (SIZE, SIZE) and out.dtype == np.uint8
        assert 0.0 <= t < 360.0


# ---------------------------------------------------------------- the crop

def test_crop_is_square_normalised_and_ink_bright():
    img = np.full((400, 400, 3), 230, np.uint8)
    img[180:220, 120:280] = 20                      # a dark stroke
    out = crop_symbol(img, (120, 180, 280, 220))
    assert out.shape == (SIZE, SIZE) and out.dtype == np.uint8
    # Ink is the bright end after the inversion, so the stroke outscores paper.
    assert out[SIZE // 2, SIZE // 2] > out[2, 2]


def test_crop_declines_a_degenerate_box():
    assert crop_symbol(np.zeros((50, 50, 3), np.uint8), (10, 10, 11, 11)) is None


# ----------------------------------------------------------------- the gate

def _term(label, x, y):
    return (label, float(x), float(y))


def _det():
    return SimpleNamespace(x1=100.0, y1=100.0, x2=160.0, y2=140.0,
                           cx=130.0, cy=120.0, w=60.0, h=40.0)


def _patch(monkeypatch, vx, vy, agree):
    import circuit2spice.diode_net as dn

    monkeypatch.setattr(dn, "predict_direction",
                        lambda *a, **k: (vx, vy, agree))


def test_terminals_are_ordered_anode_first(monkeypatch):
    """The vector points at the anode, so the terminal it points at leads."""
    terms = [_term(1, 90, 120), _term(2, 170, 120)]     # left, right
    _patch(monkeypatch, 1.0, 0.0, 1.0)                  # anode is to the right
    assert order_diode_terminals(np.zeros((5, 5, 3), np.uint8), _det(), terms) \
        == [terms[1], terms[0]]
    _patch(monkeypatch, -1.0, 0.0, 1.0)                 # anode is to the left
    assert order_diode_terminals(np.zeros((5, 5, 3), np.uint8), _det(), terms) \
        == terms


def test_an_unsure_reading_leaves_the_order_alone(monkeypatch):
    terms = [_term(1, 90, 120), _term(2, 170, 120)]
    _patch(monkeypatch, 1.0, 0.0, MIN_CONFIDENCE - 0.05)
    assert order_diode_terminals(np.zeros((5, 5, 3), np.uint8), _det(), terms) \
        is terms


def test_a_reading_across_the_terminals_is_thrown_away(monkeypatch):
    """The terminals know the axis. An answer perpendicular to it is about
    some other ink, however firmly the model stated it."""
    terms = [_term(1, 90, 120), _term(2, 170, 120)]     # axis is horizontal
    _patch(monkeypatch, 0.0, 1.0, 1.0)                  # answer is vertical
    assert order_diode_terminals(np.zeros((5, 5, 3), np.uint8), _det(), terms) \
        is terms


def test_a_missing_model_changes_nothing(monkeypatch):
    """Without weights the pipeline must behave exactly as it did before."""
    import circuit2spice.diode_net as dn

    monkeypatch.setitem(dn._CACHE, "m", None)
    terms = [_term(1, 90, 120), _term(2, 170, 120)]
    assert order_diode_terminals(np.zeros((40, 40, 3), np.uint8), _det(), terms) \
        is terms


def test_one_terminal_is_not_orderable(monkeypatch):
    _patch(monkeypatch, 1.0, 0.0, 1.0)
    terms = [_term(1, 90, 120)]
    assert order_diode_terminals(np.zeros((5, 5, 3), np.uint8), _det(), terms) \
        is terms


# ------------------------------------------------------------------ the TTA

@pytest.mark.parametrize("bearing", [0.0, 37.0, 90.0, 211.0, 300.0])
def test_tta_folds_consistent_views_back_to_the_same_bearing(bearing):
    """A perfect model, shown four turned views, must come back where it started.

    A view turned by ``+a`` sees a target at ``bearing - a``, so this feeds
    exactly that and asserts the fold-back recovers ``bearing`` with full
    agreement. It is the one place a sign slip would otherwise hide, since
    training and inference share this code and would be wrong together.
    """
    from circuit2spice.diode_net import _TTA, combine_tta

    views = [(math.cos(math.radians(bearing - a)),
              math.sin(math.radians(bearing - a))) for a in _TTA]
    got, agree = combine_tta(views)
    assert abs((got - bearing + 180) % 360 - 180) < 1e-6
    assert agree == pytest.approx(1.0, abs=1e-9)


def _views(bearing, jitter=(0, 0, 0, 0)):
    """What a model would output on each rotated view, off by ``jitter``."""
    from circuit2spice.diode_net import _TTA

    return [(math.cos(math.radians(bearing - a + j)),
             math.sin(math.radians(bearing - a + j)))
            for a, j in zip(_TTA, jitter)]


def test_tta_declines_when_the_views_cancel():
    """Four views pointing four ways is no answer, and must not become one.

    Note this is *raw* output that happens to be identical on every view --
    which, once each is turned back by its own angle, is four different
    bearings 90 degrees apart. Agreement is the right word for it.
    """
    from circuit2spice.diode_net import combine_tta

    assert combine_tta([(1, 0), (1, 0), (1, 0), (1, 0)]) is None


def test_tta_agreement_falls_as_the_views_spread():
    from circuit2spice.diode_net import combine_tta

    tight = combine_tta(_views(40.0, (0, 3, -3, 2)))[1]
    loose = combine_tta(_views(40.0, (0, 70, -70, 30)))[1]
    assert tight > 0.99 > loose


def test_a_model_of_the_wrong_shape_is_treated_as_missing(tmp_path, monkeypatch):
    """A stale weights file must not take the pipeline down with it.

    The previous model took a 48x48 single-channel crop. Such a file still
    loads; it fails on the first real crop instead, inside an optional stage.
    """
    import torch

    import circuit2spice.diode_net as dn

    bad = tmp_path / "wrong.ts.pt"
    torch.jit.script(torch.nn.Conv2d(3, 7, 1)).save(str(bad))
    dn._CACHE.pop("m", None)
    monkeypatch.setenv("CIRCUIT2SPICE_DIODE", str(bad))
    try:
        assert dn._load() is None
    finally:
        dn._CACHE.pop("m", None)
