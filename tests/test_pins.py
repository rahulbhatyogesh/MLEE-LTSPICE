"""Reading collector/base/emitter out of the drawn symbol."""

from __future__ import annotations

import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")

from circuit2spice.graph import Detection
from circuit2spice.netcompare import pin_role_accuracy
from circuit2spice.pins import order_transistor_terminals

BOX = (60, 40, 140, 160)
DET = Detection("transistor_bjt", *BOX)
BASE = (1, 55.0, 100.0)
COLLECTOR = (2, 125.0, 35.0)
EMITTER = (3, 125.0, 165.0)


def _symbol(arrowhead: bool) -> np.ndarray:
    """An NPN: vertical bar, base in from the left, two slanted leads."""
    img = np.zeros((220, 220), np.uint8)
    cv2.line(img, (85, 70), (85, 130), 255, 3)          # the bar
    cv2.line(img, (40, 100), (85, 100), 255, 3)         # base
    cv2.line(img, (85, 75), (130, 30), 255, 3)          # collector
    cv2.line(img, (85, 125), (130, 170), 255, 3)        # emitter
    if arrowhead:
        cv2.fillPoly(img, [np.array([[104, 138], [120, 148], [108, 154]])], 255)
    return img


def test_the_arrowhead_picks_out_the_emitter():
    out = order_transistor_terminals(_symbol(True), DET, [COLLECTOR, BASE, EMITTER])
    assert out == [COLLECTOR, BASE, EMITTER]


def test_the_order_it_is_given_does_not_decide_the_answer():
    """A scrambled input must come back in C-B-E order, not merely unchanged."""
    out = order_transistor_terminals(_symbol(True), DET, [EMITTER, COLLECTOR, BASE])
    assert out == [COLLECTOR, BASE, EMITTER]


def test_a_symbol_with_no_arrowhead_is_left_alone():
    """Better an order known to be arbitrary than one confidently wrong."""
    terms = [EMITTER, COLLECTOR, BASE]
    assert order_transistor_terminals(_symbol(False), DET, terms) == terms


def test_two_terminals_are_left_alone():
    terms = [COLLECTOR, EMITTER]
    assert order_transistor_terminals(_symbol(True), DET, terms) == terms


def test_no_image_is_left_alone():
    terms = [EMITTER, COLLECTOR, BASE]
    assert order_transistor_terminals(None, DET, terms) == terms


def test_pin_role_accuracy_sees_a_swap_that_exact_match_cannot():
    """The headline metric is blind here; this is what measures it."""
    ref = {"Q1": ["c", "b", "e"], "R1": ["c", "vcc"], "R2": ["e", "0"]}
    right = {"Q1": ["N2", "N1", "N3"], "R1": ["N2", "N4"], "R2": ["N3", "0"]}
    swapped = {"Q1": ["N3", "N1", "N2"], "R1": ["N2", "N4"], "R2": ["N3", "0"]}
    assert pin_role_accuracy(right, ref) == (1, 1)
    assert pin_role_accuracy(swapped, ref) == (0, 1)


def test_pin_role_accuracy_abstains_when_the_graphs_differ():
    """With no isomorphism there is no net correspondence, so nothing to score."""
    ref = {"Q1": ["c", "b", "e"], "R1": ["c", "vcc"]}
    assert pin_role_accuracy({"Q1": ["a", "b", "c"]}, ref) == (0, 0)
