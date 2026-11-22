"""Op-amp pin roles: ``invin``, ``noninvin``, ``out``.

An op-amp wired through the wrong pins opens in LTspice, round-trips, and
simulates. It is also, if the inputs are swapped, a latch rather than an
amplifier. Nothing downstream can see the difference -- graph isomorphism is
undirected -- so these tests are the only thing standing between a confident
netlist and a wrong one.

The drawings here are synthetic on purpose: a triangle, three leads and two
glyphs, with the answer known by construction, so a failure says which step
broke rather than that some photograph came out badly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from circuit2spice import opamp as oa
from circuit2spice.graph import Detection
from circuit2spice.ltspice import parse_netlist
from circuit2spice.symbols import load_symbol

# Geometry of the synthetic symbol, in image pixels (y down).
BOX = (20, 20, 120, 120)
APEX = (120, 70)
TOP_LEAD = (18.0, 45.0)        # upper input terminal, just outside the box
BOT_LEAD = (18.0, 95.0)        # lower input terminal
OUT_LEAD = (122.0, 70.0)


def _draw(top: str = "-", bottom: str = "+", glyphs: bool = True):
    """A right-pointing op-amp. Returns ``(binary, labels, det, terms)``.

    Labels: 1 on the upper input lead, 2 on the lower, 3 on the output.
    """
    img = np.zeros((160, 160), np.uint8)
    x1, y1, x2, _y2 = BOX
    cv2.line(img, (x1, y1), (x1, 120), 255, 2)          # the base
    cv2.line(img, (x1, y1), APEX, 255, 2)               # upper edge
    cv2.line(img, (x1, 120), APEX, 255, 2)              # lower edge

    def glyph(kind, cy):
        cv2.line(img, (30, cy), (42, cy), 255, 2)
        if kind == "+":
            cv2.line(img, (36, cy - 6), (36, cy + 6), 255, 2)

    if glyphs:
        glyph(top, 45)
        glyph(bottom, 95)

    labels = np.zeros_like(img, np.int32)
    cv2.line(labels, (0, 45), (x1, 45), 1, 3)
    cv2.line(labels, (0, 95), (x1, 95), 2, 3)
    cv2.line(labels, (x2, 70), (158, 70), 3, 3)

    det = Detection("opamp", *BOX)
    terms = [(1, *TOP_LEAD), (2, *BOT_LEAD), (3, *OUT_LEAD)]
    return img, labels, det, terms


def _rot(k: int, binary, labels, det, terms):
    """Turn the whole scene a quarter turn ``k`` times, terminals included."""
    h, w = binary.shape[:2]
    for _ in range(k % 4):
        binary = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
        labels = cv2.rotate(labels, cv2.ROTATE_90_CLOCKWISE)
        # (x, y) -> (h - 1 - y, x) for a clockwise quarter turn.
        def t(x, y):
            return float(h - 1 - y), float(x)
        xs_ys = [t(d.x1, d.y1) for d in ()]            # keep flake quiet
        (ax, ay), (bx, by) = t(det.x1, det.y1), t(det.x2, det.y2)
        det = Detection("opamp", min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        terms = [(lab, *t(x, y)) for lab, x, y in terms]
        h, w = w, h
        del xs_ys
    return binary, labels, det, terms


# ------------------------------------------------------------ the convention

def test_ltspice_netlists_an_opamp_as_invin_noninvin_out():
    """The order every other test here depends on.

    Confirmed against LTspice by netlisting a symbol with a distinct net
    flagged on each pin::

        X§X1 AAA BBB CCC opamp ... ;§pnba invin)noninvin)out

    Several sources give it as "In+ In- Out". Believing them swaps the inputs,
    which turns negative feedback into positive.
    """
    assert [p.name for p in load_symbol("OpAmps\\opamp").pins] == [
        "invin", "noninvin", "out"]
    assert oa.ROLES == ("invin", "noninvin", "out")


def test_ltspices_section_sign_prefix_is_stripped_from_the_ref():
    """``X§X1`` is our ``X1``, and failing to say so broke every round-trip."""
    got = parse_netlist("X§X1 N003 0 N004 opamp Aol=100K ;§pnba invin)noninvin)out")
    assert got == {"X1": ["N003", "0", "N004"]}


def test_an_ordinary_ref_is_untouched():
    assert parse_netlist("R1 a b 1k") == {"R1": ["a", "b"]}


# -------------------------------------------------------------- the geometry

def test_the_lone_terminal_is_the_output():
    _b, _l, det, terms = _draw()
    assert oa._direction_from_terminals(det, terms) == (1.0, 0.0)


def test_three_terminals_evenly_spread_decide_nothing():
    """Without a clean 2-vs-1 split there is no output to pick."""
    det = Detection("opamp", 20, 20, 120, 120)
    terms = [(1, 18.0, 45.0), (2, 70.0, 18.0), (3, 122.0, 95.0)]
    assert oa._direction_from_terminals(det, terms) is None


def test_ink_mass_finds_the_base_without_any_terminals():
    binary, _l, det, _t = _draw()
    assert oa._direction_from_ink(binary, det) == (1.0, 0.0)
    assert oa._mass_asymmetry(binary, det, 0) > oa.MIN_MASS_ASYM


def test_a_blank_box_says_nothing():
    det = Detection("opamp", 20, 20, 120, 120)
    assert oa._direction_from_ink(np.zeros((160, 160), np.uint8), det) is None


# ------------------------------------------------------------- the whole job

def test_the_signs_decide_which_input_is_which():
    binary, labels, det, terms = _draw(top="-", bottom="+")
    info: dict = {}
    got = oa.order_opamp_terminals(binary, labels, det, terms, info=info)
    assert info["state"] == "read"
    assert [g[0] for g in got] == [1, 2, 3]          # inv=top, noninv=bottom


def test_swapping_the_signs_swaps_the_inputs():
    """The point of reading them: position alone would give the same answer."""
    binary, labels, det, terms = _draw(top="+", bottom="-")
    got = oa.order_opamp_terminals(binary, labels, det, terms)
    assert [g[0] for g in got] == [2, 1, 3]
    assert got[2][0] == 3, "the output must not move when the inputs swap"


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_the_reading_survives_a_quarter_turn(k):
    """A drafter may point the triangle any of four ways."""
    scene = _rot(k, *_draw(top="-", bottom="+"))
    binary, labels, det, terms = scene
    got = oa.order_opamp_terminals(binary, labels, det, terms)
    assert [g[0] for g in got] == [1, 2, 3], f"quarter turns = {k}"


def test_the_result_is_always_a_permutation_of_the_input():
    for top, bottom in (("-", "+"), ("+", "-")):
        binary, labels, det, terms = _draw(top=top, bottom=bottom)
        got = oa.order_opamp_terminals(binary, labels, det, terms)
        assert sorted(g[0] for g in got) == sorted(t[0] for t in terms)


# --------------------------------------------------- the unity-gain buffer

def _buffer():
    """Output shorted to the inverting input: one net arrives at two pins."""
    binary, labels, det, terms = _draw(top="-", bottom="+")
    labels[labels == 1] = 3                       # upper input is the output net
    # What ``_terminals`` would hand us: two distinct labels, and the merged
    # one's centroid is the mean of two contacts on opposite sides, so it
    # points at neither pin.
    merged = (3, (TOP_LEAD[0] + OUT_LEAD[0]) / 2, (TOP_LEAD[1] + OUT_LEAD[1]) / 2)
    return binary, labels, det, [merged, (2, *BOT_LEAD)]


def test_a_buffer_recovers_the_third_pin_from_the_symbol():
    binary, labels, det, terms = _buffer()
    info: dict = {}
    got = oa.order_opamp_terminals(binary, labels, det, terms, info=info)
    assert info["state"] == "read_shared_from_ink"
    assert [g[0] for g in got] == [3, 2, 3]       # inv=out=net 3, noninv=net 2


def test_a_buffer_without_a_node_map_declines_rather_than_guesses():
    binary, _labels, det, terms = _buffer()
    got = oa.order_opamp_terminals(binary, None, det, terms)
    assert got == terms


# ------------------------------------------------------------- the declines

def test_no_signs_drawn_means_no_claim():
    binary, labels, det, terms = _draw(glyphs=False)
    info: dict = {}
    assert oa.order_opamp_terminals(binary, labels, det, terms, info=info) == terms
    assert info["state"] == "no_signs"


def test_two_of_the_same_sign_means_no_claim():
    """A stray dash beside each input is not evidence, and must not read as
    one. The test is the *difference* between the sides for exactly this."""
    binary, labels, det, terms = _draw(top="-", bottom="-")
    info: dict = {}
    assert oa.order_opamp_terminals(binary, labels, det, terms, info=info) == terms
    assert info["state"] == "no_signs"


def test_terminals_and_ink_pointing_opposite_ways_means_no_claim():
    binary, labels, det, terms = _draw()
    flipped = [(lab, 2 * det.cx - x, y) for lab, x, y in terms]
    info: dict = {}
    assert oa.order_opamp_terminals(binary, labels, det, flipped, info=info) == flipped
    assert info["state"] == "direction_conflict"


def test_two_terminals_and_no_ink_at_all_means_no_claim():
    _b, labels, det, terms = _buffer()
    info: dict = {}
    assert oa.order_opamp_terminals(None, labels, det, terms, info=info) == terms
    assert info["state"] == "no_direction"


def test_a_tiny_box_is_not_read():
    det = Detection("opamp", 0, 0, 6, 6)
    assert oa._mass_asymmetry(np.ones((20, 20), np.uint8), det, 0) is None


# --------------- a sign fused to its own lead: a known, declared limit

def _fused(top="-", bottom="+"):
    """The glyph drawn collinear with the lead and touching the symbol edge,
    which is what a cleanly drawn schematic does -- and which defeats a
    connected-component search, because the whole thing reaches the border."""
    img = np.zeros((160, 160), np.uint8)
    x1, y1, x2, _ = BOX
    cv2.line(img, (x1, y1), (x1, 120), 255, 2)
    cv2.line(img, (x1, y1), APEX, 255, 2)
    cv2.line(img, (x1, 120), APEX, 255, 2)

    def lead_and_glyph(kind, cy):
        cv2.line(img, (0, cy), (44, cy), 255, 2)      # lead, through the base
        if kind == "+":
            cv2.line(img, (38, cy - 7), (38, cy + 7), 255, 2)

    lead_and_glyph(top, 45)
    lead_and_glyph(bottom, 95)

    labels = np.zeros_like(img, np.int32)
    cv2.line(labels, (0, 45), (x1, 45), 1, 3)
    cv2.line(labels, (0, 95), (x1, 95), 2, 3)
    cv2.line(labels, (x2, 70), (158, 70), 3, 3)
    det = Detection("opamp", *BOX)
    return img, labels, det, [(1, *TOP_LEAD), (2, *BOT_LEAD), (3, *OUT_LEAD)]


def test_a_sign_fused_to_its_lead_declines_rather_than_guessing():
    """The limit, pinned so it is a decision and not an accident.

    A shape-based fallback (`pins.sign_by_cross_stroke`) was written for
    exactly this case and measured *inverted* on both real op-amps that
    needed it -- the window picks up the triangle's slanted edge, not the
    glyph. Declining leaves the terminals in a known-arbitrary order, which
    the caller is told about; guessing does not.
    """
    binary, labels, det, terms = _fused(top="-", bottom="+")
    info: dict = {}
    assert oa.order_opamp_terminals(binary, labels, det, terms, info=info) == terms
    assert info["state"] == "no_signs"
