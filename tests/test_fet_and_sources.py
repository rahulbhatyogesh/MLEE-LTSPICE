"""Reading a FET's gate and a source's polarity off the symbol.

Both were ordered by skeleton pixel count before this, which is no order at
all, and both fail the same silent way: the netlist parses, LTspice opens it,
the round-trip passes, and graph isomorphism cannot see a pin role. A supply
the wrong way round inverts every node voltage that depends on it.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from circuit2spice.graph import Detection
from circuit2spice.pins import order_fet_terminals, order_source_terminals


# ----------------------------------------------------------------- the FET

FET = Detection("transistor_fet", 20, 20, 120, 120)


def _fet(gate, a, b):
    """terms with labels: 7 = gate, 5 = the 'a' lead, 0 = the 'b' lead."""
    return [(7, *gate), (5, *a), (0, *b)]


def test_the_gate_is_the_lone_lead():
    got = order_fet_terminals(FET, _fet((18., 70.), (100., 18.), (100., 122.)))
    assert got[1][0] == 7, "the gate must be SPICE's middle node"


def test_drain_and_source_follow_the_drawn_convention():
    """Gate left, so drain is up -- CGHD's own port table, drain [0.5, 0]
    above source [0.5, 1]."""
    got = order_fet_terminals(FET, _fet((18., 70.), (100., 18.), (100., 122.)))
    assert [t[0] for t in got] == [5, 7, 0]


def test_the_convention_turns_with_the_symbol():
    # Gate at the top: a quarter turn puts the drain to the right.
    got = order_fet_terminals(FET, _fet((70., 18.), (122., 100.), (18., 100.)))
    assert [t[0] for t in got] == [5, 7, 0]


def test_a_mirrored_fet_comes_out_reversed_and_that_is_documented():
    """Not a bug to fix here -- the symbol carries no mark that distinguishes
    drain from source, so a mirrored one is a coin toss `--flip` settles."""
    got = order_fet_terminals(FET, _fet((122., 70.), (40., 18.), (40., 122.)))
    assert [t[0] for t in got] == [0, 7, 5]


def test_the_gate_still_lands_in_the_middle_when_mirrored():
    """Which is the point: `--flip` swaps the outer two, so a single flag
    fixes a reversed FET only if the gate is already central."""
    got = order_fet_terminals(FET, _fet((122., 70.), (40., 18.), (40., 122.)))
    assert got[1][0] == 7


def test_three_leads_spread_evenly_are_not_a_gate_split():
    info: dict = {}
    terms = [(7, 18., 45.), (5, 70., 18.), (0, 122., 95.)]
    assert order_fet_terminals(FET, terms, info) == terms
    assert info["state"] == "no_gate"


def test_two_terminals_are_left_alone():
    info: dict = {}
    terms = [(7, 18., 70.), (5, 122., 70.)]
    assert order_fet_terminals(FET, terms, info) == terms
    assert info["state"] == "not_three_terminals"


# ------------------------------------------------------------- the sources

SRC = Detection("source_voltage", 30, 30, 130, 130)


def _source(plus_on_top=True):
    """A circle with a + and a - inside, and a lead out of each end."""
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (80, 80), 48, 255, 3)
    top, bot = (("+", "-") if plus_on_top else ("-", "+"))

    def glyph(kind, cy):
        cv2.line(img, (68, cy), (92, cy), 255, 3)
        if kind == "+":
            cv2.line(img, (80, cy - 12), (80, cy + 12), 255, 3)

    glyph(top, 56)
    glyph(bot, 108)
    # terms[0] is the upper lead, terms[1] the lower
    return img, [(1, 80.0, 26.0), (2, 80.0, 134.0)]


def test_the_plus_terminal_comes_first():
    img, terms = _source(plus_on_top=True)
    info: dict = {}
    got = order_source_terminals(img, SRC, terms, info)
    assert info["state"] == "read"
    assert [t[0] for t in got] == [1, 2]


def test_turning_the_source_over_turns_the_netlist_over():
    img, terms = _source(plus_on_top=False)
    got = order_source_terminals(img, SRC, terms)
    assert [t[0] for t in got] == [2, 1]


def test_no_signs_drawn_means_no_claim():
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (80, 80), 48, 255, 3)
    terms = [(1, 80.0, 26.0), (2, 80.0, 134.0)]
    info: dict = {}
    assert order_source_terminals(img, SRC, terms, info) == terms
    assert info["state"] in ("no_signs", "one_sided")


def test_two_dashes_are_not_a_polarity():
    """A stray mark beside each terminal must not read as a sign."""
    img = np.zeros((200, 200), np.uint8)
    cv2.circle(img, (80, 80), 48, 255, 3)
    for cy in (56, 108):
        cv2.line(img, (68, cy), (92, cy), 255, 3)
    terms = [(1, 80.0, 26.0), (2, 80.0, 134.0)]
    info: dict = {}
    assert order_source_terminals(img, SRC, terms, info) == terms
    assert info["state"] == "signs_alike"


def test_no_ink_at_all_means_no_claim():
    _img, terms = _source()
    assert order_source_terminals(None, SRC, terms) == terms


@pytest.mark.parametrize("n", [1, 3])
def test_only_a_two_terminal_part_is_read(n):
    img, terms = _source()
    short = terms[:1] * n
    assert order_source_terminals(img, SRC, short) == short
