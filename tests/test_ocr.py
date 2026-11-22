"""The closed vocabulary and the priors that narrow it.

These cover everything in :mod:`circuit2spice.ocr` that does not need the
recogniser, which is where the measured failures actually came from.
"""

from __future__ import annotations

import pytest

from circuit2spice.graph import Detection
from circuit2spice.ocr import (
    NET_LABELS, beside_a_terminal, designators, nearest_letter,
    plausible_length, values,
)
from circuit2spice.values import parse_label


def _text(x1, y1, x2, y2):
    return Detection("text", x1, y1, x2, y2, 1.0)


def test_every_candidate_is_a_legal_label():
    """The point of a closed vocabulary: the output cannot be unparseable."""
    for letter in "RCLVI":
        for cand in designators(letter, with_nets=False) + values(letter):
            d, v = parse_label(cand)
            assert d or v, cand


def test_the_unit_offered_matches_the_class():
    assert all(c.endswith(("Ω", "ohms", "ohm")) for c in values("R"))
    assert all(parse_label(c)[1].unit == "F" for c in values("C"))
    assert all(parse_label(c)[1].unit == "H" for c in values("L"))


def test_net_labels_are_available_only_where_a_net_label_belongs():
    """Regression: Vout sits nearest a capacitor.

    A pool built from that capacitor's class cannot contain Vout, so the
    scorer has to answer with a capacitance. Offering net labels *everywhere*
    is the opposite failure -- Rout and VDD then win boxes holding a resistor
    value.
    """
    assert set(NET_LABELS) <= set(designators("C", with_nets=True))
    assert not set(NET_LABELS) & set(designators("C", with_nets=False))


def test_a_label_by_a_terminal_may_name_a_net():
    label = _text(340, 100, 400, 130)
    term = Detection("terminal", 300, 100, 330, 130)
    far = Detection("resistor", 900, 100, 940, 180)
    assert beside_a_terminal(label, [label, term, far])


def test_a_label_by_a_part_may_not():
    label = _text(340, 100, 400, 130)
    res = Detection("resistor", 300, 100, 330, 180)
    term = Detection("terminal", 900, 100, 930, 130)
    assert not beside_a_terminal(label, [label, res, term])


@pytest.mark.parametrize("w, h, truth", [
    (61, 57, "R2"), (50, 58, "C1"), (41, 39, "Vin"), (106, 59, "Cout"),
    (85, 47, "10Ω"), (93, 69, "1pF"), (55, 58, "R5"),
    (156, 55, "R2 10Ω"), (160, 68, "R1 1kΩ"), (202, 69, "C2 10F"),
    (141, 61, "1000F"), (168, 63, "R3 10Ω"), (155, 89, "R4 1Ω"),
])
def test_the_length_prior_keeps_the_true_label(w, h, truth):
    """Measured on a real drawing: all 13 survive, so the prior is safe."""
    det = _text(0, 0, w, h)
    assert truth in plausible_length(det, [truth, "Rout", "Cout", "4700pF"])


def test_the_length_prior_drops_a_decoy_too_long_for_the_box():
    """Rout kept winning narrow boxes: a mean log-prob likes short strings."""
    narrow = _text(0, 0, 61, 57)          # the box that really says "R2"
    assert "4700pF" not in plausible_length(narrow, ["R2", "4700pF"])


def test_the_length_prior_never_empties_the_pool():
    """A bad estimate must not silence a box altogether."""
    weird = _text(0, 0, 4000, 3)
    assert plausible_length(weird, ["R1", "C2"]) == ["R1", "C2"]


def test_nearest_letter_follows_the_detector():
    label = _text(200, 0, 260, 30)
    cap = Detection("capacitor", 150, 0, 190, 40)
    res = Detection("resistor", 900, 0, 940, 80)
    assert nearest_letter(label, [cap, res]) == "C"


# ---------------------------------------------------- repairing a reading

@pytest.mark.parametrize("raw, letter, want", [
    ("10r", "R", "10\u03a9"),          # omega read as r
    ("6n", "R", "6\u03a9"),            # omega read as n
    # ``R410`` is the omega read as a 0. The designator is recovered; the
    # magnitude is not, and cannot be: ``R41\u03a9`` is the same string shape as
    # ``R15\u03a9``, which is a two-digit designator whose value went unread.
    # Splitting one means splitting the other, and splitting ``R15\u03a9`` into
    # R1 + 5 ohms is how eight resistors on a 40-part drawing came back
    # holding 0..7 ohms. Preferring the designator loses an order of magnitude
    # here and costs nothing on the 15-label set, still measured at 14/15.
    ("R410", "R", "R4 10\u03a9"),
    ("R3102", "R", "R3 10\u03a9"),     # omega read as 2
    ("Rs", "R", "R5"),                 # 5 read as s
])
def test_a_mangled_unit_is_repaired(raw, letter, want):
    """Every residual error on the test drawing was one of these five."""
    from circuit2spice.ocr import repair_reading

    assert parse_label(repair_reading(raw, letter)) [0] == parse_label(want)[0]
    got_v = parse_label(repair_reading(raw, letter))[1]
    want_v = parse_label(want)[1]
    assert (got_v.spice if got_v else None) == (want_v.spice if want_v else None)


def test_repair_prefers_a_single_digit_index():
    """Regression: R3102 already parses, as R310 with the value 2.

    A first-match rule therefore keeps the wrong reading, which is why the
    candidates are ranked instead of taken in order.
    """
    from circuit2spice.ocr import repair_reading

    assert parse_label(repair_reading("R3102", "R"))[0] == "R3"


@pytest.mark.parametrize("good", ["Cout", "R2", "C1", "Vout", "C210F", "1pf"])
def test_repair_leaves_a_good_reading_alone(good):
    from circuit2spice.ocr import repair_reading

    before = parse_label(good)
    after = parse_label(repair_reading(good, "C"))
    assert before[0] == after[0]
    assert (before[1].spice if before[1] else None) == (after[1].spice if after[1] else None)


def test_repair_of_nonsense_is_still_nonsense_not_an_invention():
    """It must not manufacture a label out of an unreadable box."""
    from circuit2spice.ocr import repair_reading

    out = repair_reading("", "R")
    assert out == ""


# -------------------------------------------------- reading through Ollama

def test_a_thinking_model_s_reasoning_is_not_the_answer():
    """``qwen3-vl:4b-thinking`` narrates first; the label is the last line."""
    from circuit2spice.ocr import clean_reading

    assert clean_reading("<think>could be R1 or RI</think>\nR1 1k") == "R1 1k"


def test_a_sentence_around_the_answer_is_stripped():
    from circuit2spice.ocr import clean_reading

    assert clean_reading("The label reads: R1 1k.") == "R1 1k"
    assert clean_reading("```\nCout\n```") == "Cout"


def test_a_decimal_point_survives_the_cleaning():
    """Trailing full stops go; a decimal separator is not one."""
    from circuit2spice.ocr import clean_reading

    assert clean_reading("R1 2.2k.") == "R1 2.2k"


def test_cleaning_nothing_invents_nothing():
    from circuit2spice.ocr import clean_reading

    assert clean_reading("   \n\n") == ""
    assert clean_reading("<think>I cannot read it</think>") == ""


def test_the_request_asks_for_one_deterministic_answer():
    """Sampling is how a reader invents a *plausible* wrong label."""
    import base64

    import cv2
    import numpy as np

    from circuit2spice.ocr import ollama_payload

    crop = np.full((12, 30), 200, np.uint8)
    body = ollama_payload("qwen3-vl:4b", "read it", crop)
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.0
    back = cv2.imdecode(np.frombuffer(base64.b64decode(body["images"][0]),
                                      np.uint8), cv2.IMREAD_GRAYSCALE)
    assert back.shape == crop.shape          # the crop survives the round trip


def test_no_service_is_a_silent_no_not_a_crash():
    """The normal case on a machine without Ollama, so it must cost nothing."""
    from circuit2spice.ocr import make_reader, ollama_models

    dead = "http://127.0.0.1:1"
    assert ollama_models(dead, timeout=0.2) == []
    import circuit2spice.ocr as ocr_mod
    old = ocr_mod.OLLAMA_URL
    try:
        ocr_mod.OLLAMA_URL = dead
        assert make_reader("ollama") is None
    finally:
        ocr_mod.OLLAMA_URL = old


@pytest.mark.parametrize("raw, want_value", [
    ("R4 1n", "1"),      # one nanoohm is not a part anyone draws
    ("6n", "6"),
])
def test_a_sub_ohm_resistor_loses_to_the_omega_it_misread(raw, want_value):
    """``R4 1n`` parses perfectly well, which is why it needed a prior.

    Measured on the 4-bit GPU readers: both return ``1n`` and ``6n`` where the
    drawing says ohms. Every other term in ``_plausibility`` ties, so without
    a magnitude prior the misreading wins on being first.
    """
    from circuit2spice.ocr import repair_reading

    value = parse_label(repair_reading(raw, "R"))[1]
    assert value is not None and value.spice == want_value
    assert value.unit == "ohm"


def test_an_absurd_capacitor_is_still_what_the_drawing_says():
    """The prior is resistance-only on purpose: `C2 10F` means ten farads."""
    from circuit2spice.ocr import repair_reading

    assert parse_label(repair_reading("C2 10F", "C"))[1].spice == "10"
    assert parse_label(repair_reading("1pF", "C"))[1].spice == "1p"


def test_a_unit_that_arrived_as_junk_is_trimmed():
    """The 8-bit reader expands the omega instead of substituting it."""
    from circuit2spice.ocr import repair_reading

    desig, value = parse_label(repair_reading("R1 1k-52", "R"))
    assert desig == "R1"
    assert value is not None and value.spice == "1k"


def test_trimming_cannot_beat_a_reading_that_already_parses():
    """Otherwise every long label would be eaten one character at a time."""
    from circuit2spice.ocr import repair_reading

    for good in ["R2 10\u03a9", "C2 10F", "1000F", "Cout", "R1 2.2k\u03a9"]:
        assert repair_reading(good, good[0] if good[0].isalpha() else "C") == good


def test_a_class_letter_lost_to_handwriting_still_yields_its_value():
    """`L1 3H` is written with the L and 1 joined; readers return `U3H`.

    The index is genuinely illegible, but the henries are not, and the value
    is the half that decides whether the circuit simulates. The designator
    falls back to the positional one.
    """
    from circuit2spice.ocr import repair_reading

    desig, value = parse_label(repair_reading("U3H", "L"))
    assert desig is None
    assert value is not None and value.spice == "3" and value.unit == "H"


def test_dropping_a_leading_letter_cannot_eat_a_designator():
    """`R5` must not become the value 5: ranking has to refuse the trim."""
    from circuit2spice.ocr import repair_reading

    for good in ["R5", "C1", "R4 1\u03a9", "C2 10F"]:
        assert repair_reading(good, good[0]) == good


def test_the_class_hint_never_names_the_reference_letter():
    """Naming it cost four labels: the reader prepends one that is not there."""
    from circuit2spice.ocr import prompt_for

    hint = prompt_for("R")
    assert "ohms" in hint
    assert "starts with" not in hint


def test_no_class_hint_where_a_net_label_belongs():
    """Vout's nearest component is a capacitor; the hint made it `Cout`."""
    from circuit2spice.ocr import VLM_PROMPT, prompt_for

    assert prompt_for("C", with_nets=True) == VLM_PROMPT


# ------------------------------------------- a bare value keeps its prefix

@pytest.mark.parametrize("raw, letter, want", [
    ("1k", "R", "1k"),       # the common case: "R1" on one line, "1k" below
    ("10k", "R", "10k"),
    ("1m", "L", "1m"),       # milli-henries, not one henry
    ("1u", "C", "1u"),
    ("2.2k", "R", "2.2k"),
])
def test_a_bare_value_survives_the_repair(raw, letter, want):
    """Regression, and an expensive one: every unitless value was destroyed.

    A drawing that writes ``R1`` above ``1k`` gives the reader a box holding
    only ``1k``. That parses as nothing without a unit, so it scored ``-1e9``
    and lost to candidates built by *deleting* its last character and
    appending the unit -- ``1k`` came back as ``1Ω``, ``10k`` as ``10Ω``,
    ``1m`` as ``1Ω``. The reader had read all of them correctly.

    Nothing downstream could catch it: ``1Ω`` is a perfectly legal label.
    """
    from circuit2spice.ocr import repair_reading

    repaired = repair_reading(raw, letter)
    assert repaired == want, f"{raw!r} was rewritten as {repaired!r}"


@pytest.mark.parametrize("raw, letter, want", [
    ("1k", "R", "1k\u03a9"),
    ("1m", "L", "1mH"),
    ("1u", "C", "1uF"),
    ("R1 1k", "R", "R1 1k"),     # already complete: left alone
    ("R5", "R", "R5"),           # a designator is not a value
    ("", "R", ""),
    ("1k", None, "1k"),          # no neighbouring class: assume nothing
])
def test_the_implied_unit_is_written_onto_a_bare_value(raw, letter, want):
    """Downstream keys off the unit, so a value without one is dropped.

    ``values.py`` picks a component for a value by what the unit *could*
    belong to. A bare ``1k`` names no class, so it was assigned to nothing
    and the part kept its default -- the value was read correctly and then
    thrown away.
    """
    from circuit2spice.ocr import complete_unit

    assert complete_unit(raw, letter) == want


def test_an_unrecognised_trailing_glyph_still_fails_to_parse():
    """The guard that keeps the repair working.

    ``10r`` must not parse as "10, assume ohms, ignore the r": if it does, a
    wrong reading scores as well as the right one and ``10Ω`` is never
    preferred. Equally ``1k`` must parse, or we are back to the bug above.
    """
    assert parse_label("10r", assume_unit="ohm") == (None, None)
    _d, v = parse_label("1k", assume_unit="ohm")
    assert v is not None and v.spice == "1k" and v.unit == "ohm"


def test_the_raw_reading_is_kept_so_a_re_type_can_redo_the_repair():
    """``repair_reading`` is unit-dependent and lossy. Beside a resistor the
    drawing's ``100u`` becomes ``100Ω`` -- a sub-microohm resistor being
    implausible -- and the microhenries cannot be got back out of that. So
    correcting the class has to re-repair from the raw string."""
    from circuit2spice.graph import Detection
    from circuit2spice.ocr import rerepair

    text = Detection("text", 10, 0, 40, 14)
    text.text_raw, text.text = "100u", "100\u03a9"
    part = Detection("inductor", 8, 20, 48, 60)       # re-typed since the read
    assert rerepair([text, part]) == 1
    assert text.text == "100uH"


def test_rerepair_leaves_a_reading_alone_when_nothing_moved():
    from circuit2spice.graph import Detection
    from circuit2spice.ocr import rerepair

    text = Detection("text", 10, 0, 40, 14)
    text.text_raw, text.text = "100u", "100\u03a9"
    part = Detection("resistor", 8, 20, 48, 60)        # class unchanged
    assert rerepair([text, part]) == 0
    assert text.text == "100\u03a9"


def test_rerepair_without_a_raw_reading_does_nothing():
    from circuit2spice.graph import Detection
    from circuit2spice.ocr import rerepair

    text = Detection("text", 10, 0, 40, 14, text="R1")
    assert rerepair([text, Detection("resistor", 8, 20, 48, 60)]) == 0


@pytest.mark.parametrize("raw, letter, want", [
    ("-10n", "C", "10n"),     # the capacitor's own plate, read as a hyphen
    ("-10u", "C", "10u"),
    ("-1u", "C", "1u"),
    (".1k", "R", "1k"),       # ...and as a full stop
    ("-10k", "R", "10k"),
    ("-C1", "C", "C1"),       # a designator loses it too
    ("-R5", "R", "R5"),
])
def test_a_stroke_caught_in_the_crop_does_not_eat_the_value(raw, letter, want):
    """Regression. ``_VALUE_RE`` anchors on a digit, so a reading that starts
    with punctuation parses as *nothing* -- and a value that parses as nothing
    is dropped, leaving the part on its class default with no trace that
    anything was read. Three labels on one drawing went that way: the
    capacitor plate came back as a leading hyphen on both of its values.
    """
    from circuit2spice.ocr import repair_reading

    assert repair_reading(raw, letter) == want


def test_stripping_punctuation_cannot_rescue_a_reading_that_still_fails():
    """The strip only ever *adds* a candidate; ranking still has to accept
    it, so nothing that fails to parse can win by being stripped."""
    from circuit2spice.ocr import repair_reading

    assert repair_reading("---", "R") == "---"
