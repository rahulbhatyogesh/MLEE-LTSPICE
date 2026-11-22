"""Reading handwritten component values."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from circuit2spice.graph import Detection
from circuit2spice.netcompare import informative_values, value_f1
from circuit2spice.values import (
    apply_names_to_graph, assign_labels, name_nets_from_labels, parse_label,
    parse_net_label, parse_value, to_magnitude, values_equal,
    values_from_detections,
)


@pytest.mark.parametrize("raw, spice, unit", [
    ("10KΩ", "10k", "ohm"),
    ("1kΩ", "1k", "ohm"),
    ("100Ω", "100", "ohm"),
    ("2.2KΩ", "2.2k", "ohm"),
    ("2,2KΩ", "2.2k", "ohm"),        # one drafter used a comma decimal
    ("1µF", "1u", "F"),
    ("100pF", "100p", "F"),
    ("10nH", "10n", "H"),
    ("9V", "9", "V"),
    ("9v", "9", "V"),
    ("10V AC", "10", "V"),
])
def test_values_parse(raw, spice, unit):
    v = parse_value(raw)
    assert v is not None, raw
    assert (v.spice, v.unit) == (spice, unit)


@pytest.mark.parametrize("raw", ["VDD", "GND", "OUT", "TRIG", "THRS", "A", "B", "CV", ""])
def test_node_labels_are_not_values(raw):
    """A label is not a value; guessing one is worse than keeping the default."""
    assert parse_value(raw) is None


@pytest.mark.parametrize("raw", ["2N2222", "TIP41", "2N2907", "NE555"])
def test_part_numbers_are_model_names(raw):
    v = parse_value(raw)
    assert v is not None and v.is_model and v.spice == raw.upper()


def test_uppercase_M_is_mega_not_milli():
    """SPICE reads M as milli; a handwritten M means mega, so it must not pass through."""
    v = parse_value("10MΩ")
    assert v is not None
    assert v.spice == "10meg"
    assert v.magnitude == pytest.approx(10e6)


def test_unit_decides_the_component_class():
    assert parse_value("10KΩ").classes == ("resistor",)
    assert parse_value("1µF").classes == ("capacitor",)
    assert parse_value("10nH").classes == ("inductor",)


@pytest.mark.parametrize("a, b", [
    ("10k", "10000"), ("1u", "1µ"), ("1u", "1Âµ"),      # mojibake from latin-1
    ("100k", "100K tol=1 pwr=0.1"),                      # LTspice parameter tail
    ("2N2222", "2n2222"),
])
def test_equivalent_values_compare_equal(a, b):
    assert values_equal(to_magnitude(a), to_magnitude(b))


def test_different_values_do_not_compare_equal():
    assert not values_equal(to_magnitude("1u"), to_magnitude("10u"))
    assert not values_equal(to_magnitude("1k"), to_magnitude("1m"))


def _text(s, x1, y1, x2, y2):
    return Detection("text", x1, y1, x2, y2, 1.0, text=s)


def test_a_label_is_assigned_to_a_compatible_component():
    """The unit rules out the nearer capacitor: ohms can only be a resistor."""
    dets = [
        _text("10KΩ", 100, 0, 180, 30),
        Detection("capacitor", 60, 0, 95, 30),      # nearer, but wrong type
        Detection("resistor", 200, 0, 240, 40),
    ]
    got = values_from_detections(dets)
    assert got.get(id(dets[2])) == "10k"
    assert id(dets[1]) not in got


def test_a_wide_label_does_not_steal_the_component_it_merely_reaches_past():
    """Regression: edge-to-edge distance let a long label graze a neighbour.

    The label sits just right of the upper resistor, but its far end overlaps
    the box of the lower one, which scored a perfect 0 gap and took the value.
    """
    label = _text("10KΩ", 590, 519, 695, 553)
    near = Detection("resistor", 540, 494, 588, 587)    # the one it labels
    grazed = Detection("resistor", 690, 543, 739, 623)  # merely overlapped
    got = values_from_detections([label, near, grazed])
    assert got.get(id(near)) == "10k", "label went to the wrong resistor"
    assert id(grazed) not in got


def test_one_value_per_component_and_one_component_per_value():
    dets = [
        _text("10KΩ", 100, 0, 180, 30),
        _text("100Ω", 100, 200, 180, 230),
        Detection("resistor", 200, 0, 240, 40),
        Detection("resistor", 200, 200, 240, 240),
    ]
    got = values_from_detections(dets)
    assert got == {id(dets[2]): "10k", id(dets[3]): "100"}


def test_bill_of_values_ignores_instance_naming():
    """Our refs come from page position, LTspice's from the schematic."""
    ours = {"R1": "10k", "R2": "100", "C1": "1u"}
    ref = {"R7": "100", "R3": "10k", "C9": "1µ"}
    assert value_f1(ours, ref)[2] == pytest.approx(1.0)


def test_unvalued_references_are_excluded_from_scoring():
    """LTspice prints the bare symbol letter when the Value field is empty."""
    ref = {"R1": "R", "C1": "C", "L1": "L", "V1": "V", "R2": "10k"}
    assert informative_values(ref) == {"R2": "10k"}


def test_ac_specifications_are_excluded_from_scoring():
    """A drawing cannot state SINE(0 20m 400), so it is not a fair target."""
    ref = {"V1": "SINE(0 20m 400)", "V2": "AC 2", "V3": "9"}
    assert informative_values(ref) == {"V3": "9"}


# --------------------------------------------------- designators and renaming

@pytest.mark.parametrize("raw, desig, value", [
    ("R1 1k\u03a9", "R1", "1k"),
    ("R\u2081 1k\u03a9", "R1", "1k"),       # people write subscripts
    ("R1 10k", "R1", "10k"),                # the R supplies the missing unit
    ("C1 100n", "C1", "100n"),
    ("C2 10F", "C2", "10"),                 # not the part number "C210F"
    ("R110k", "R1", "10k"),                 # written with no space
    ("Cout", "Cout", None),
    ("Vin", "Vin", None),
    ("1pF", None, "1p"),
    ("Q1 2N2222", "Q1", "2N2222"),
])
def test_a_label_splits_into_designator_and_value(raw, desig, value):
    d, v = parse_label(raw)
    assert d == desig
    assert (v.spice if v else None) == value


@pytest.mark.parametrize("raw", ["VDD", "VCC", "VSS", "GND"])
def test_rail_names_are_not_designators(raw):
    """VDD names a net, not a part; letting it through renames a real source."""
    assert parse_label(raw) == (None, None)


def test_a_bare_number_still_needs_a_unit():
    """Only a designator may supply the unit; alone, 10k could be anything."""
    assert parse_label("10k") == (None, None)


def _comp(ref, det):
    return SimpleNamespace(ref=ref, det=det, cls=det.cls)


def _graph(*comps):
    return SimpleNamespace(components=list(comps))


def test_designator_and_value_from_one_box_land_on_one_component():
    """Matched separately, the name and the value can go to different parts."""
    label = _text("R1 1k\u03a9", 200, 0, 320, 30)
    near = Detection("resistor", 150, 0, 190, 80)
    far = Detection("resistor", 340, 0, 380, 80)
    values, names = assign_labels([label, near, far])
    assert values.get(id(near)) == "1k"
    assert names.get(id(near)) == "R1"
    assert id(far) not in values and id(far) not in names


def test_renaming_handles_a_straight_swap():
    """Regression: our R1 is their R2 and vice versa.

    Renamed one at a time, both are refused -- each wanted name is still held
    by the other -- and the netlist silently keeps the positional names.
    """
    d1, d2 = Detection("resistor", 0, 0, 10, 10), Detection("resistor", 0, 20, 10, 30)
    c1, c2 = _comp("R1", d1), _comp("R2", d2)
    n = apply_names_to_graph(_graph(c1, c2), {id(d1): "R2", id(d2): "R1"})
    assert (c1.ref, c2.ref) == ("R2", "R1")
    assert n == 2


def test_a_rename_may_not_change_the_element_type():
    """LTspice dispatches on the first letter: C1 -> R3 would become a resistor."""
    d = Detection("capacitor", 0, 0, 10, 10)
    c = _comp("C1", d)
    assert apply_names_to_graph(_graph(c), {id(d): "R3"}) == 0
    assert c.ref == "C1"


def test_an_unnamed_component_is_moved_off_a_claimed_name():
    """If their R1 is our R2, our R1 must vacate rather than collide."""
    d1, d2 = Detection("resistor", 0, 0, 10, 10), Detection("resistor", 0, 20, 10, 30)
    c1, c2 = _comp("R1", d1), _comp("R2", d2)
    apply_names_to_graph(_graph(c1, c2), {id(d2): "R1"})
    assert c2.ref == "R1"
    assert c1.ref != "R1"


# ------------------------------------------------------------- named nets

@pytest.mark.parametrize("raw, name", [
    ("Vout", "Vout"), ("VOUT", "Vout"), ("vout", "Vout"), ("out", "Vout"),
    ("Vin", "Vin"), ("in", "Vin"), ("gnd", "GND"), ("CLK", "CLK"),
])
def test_net_labels_normalise_to_one_spelling(raw, name):
    """VOUT and Vout must be one wire, not two."""
    assert parse_net_label(raw) == name


@pytest.mark.parametrize("raw", ["R1", "1k\u03a9", "2N2222", "10F", ""])
def test_a_value_or_designator_is_not_a_net_name(raw):
    assert parse_net_label(raw) is None


def test_a_label_beside_a_terminal_names_its_net():
    """Without this the output is N004 and a simulation has to guess."""
    term = Detection("terminal", 300, 100, 330, 130)
    label = _text("Vout", 340, 100, 400, 130)
    g = SimpleNamespace(net_names={}, ground_node=0,
                        terminal_nodes={id(term): 4})
    assert name_nets_from_labels(g, [term, label]) == 1
    assert g.net_names == {4: "Vout"}


def test_ground_is_never_renamed():
    """SPICE requires node 0; calling it GND breaks the netlist."""
    term = Detection("terminal", 300, 100, 330, 130)
    label = _text("GND", 340, 100, 400, 130)
    g = SimpleNamespace(net_names={}, ground_node=0,
                        terminal_nodes={id(term): 0})
    assert name_nets_from_labels(g, [term, label]) == 0
    assert g.net_names == {}


def test_one_name_per_net_and_the_nearer_label_wins():
    term = Detection("terminal", 300, 100, 330, 130)
    near = _text("Vout", 335, 100, 380, 130)
    far = _text("CLK", 600, 100, 650, 130)
    g = SimpleNamespace(net_names={}, ground_node=0,
                        terminal_nodes={id(term): 7})
    name_nets_from_labels(g, [term, near, far])
    assert g.net_names == {7: "Vout"}


def test_a_named_net_shows_up_in_the_netlist():
    from circuit2spice.graph import CircuitGraph
    g = CircuitGraph(n_nodes=5, ground_node=0)
    assert g.node_name(4) == "N004"
    g.net_names[4] = "Vout"
    assert g.node_name(4) == "Vout"
    assert g.node_name(0) == "0"


# -------------------------------------- what was read vs what was guessed

def test_a_default_is_distinguished_from_an_identical_reading():
    """``1k`` read off the page and ``1k`` from SPECS are the same four
    characters in the netlist, so the difference has to be recorded as it
    happens, not inferred afterwards."""
    from circuit2spice.graph import CircuitGraph, Component, Detection
    from circuit2spice.values import apply_to_graph, defaulted_components

    d1, d2 = Detection("resistor", 0, 0, 10, 10), Detection("resistor", 50, 0, 60, 10)
    g = CircuitGraph(
        components=[Component("R1", "resistor", d1, [1, 0], "1k"),
                    Component("R2", "resistor", d2, [2, 0], "1k")],
        n_nodes=2, ground_node=0)
    read: set = set()
    apply_to_graph(g, {id(d1): "1k"}, applied=read)    # R1 really was read
    assert defaulted_components(g, read_ids=read) == ["R2"]


def test_a_value_the_user_stated_is_not_called_a_guess():
    from circuit2spice.graph import CircuitGraph, Component, Detection
    from circuit2spice.values import defaulted_components

    d = Detection("capacitor", 0, 0, 10, 10)
    g = CircuitGraph(components=[Component("C1", "capacitor", d, [1, 0], "10u")],
                     n_nodes=1, ground_node=0)
    assert defaulted_components(g, read_ids=set()) == ["C1"]
    assert defaulted_components(g, read_ids=set(), explicit={"C1"}) == []


def test_a_warning_follows_a_part_that_gets_renamed():
    """Warnings are raised against positional refs and the page then permutes
    them, so "V3 has all terminals on one node" ended up accusing a healthy
    V3 while the shorted part was called V2."""
    from circuit2spice.graph import CircuitGraph, Component, Detection
    from circuit2spice.values import apply_names_to_graph

    d = Detection("source_voltage", 0, 0, 10, 10)
    g = CircuitGraph(
        components=[Component("V3", "source_voltage", d, [0, 0], "12")],
        n_nodes=1, ground_node=0,
        warnings=["V3 has all terminals on one node (short circuit)"])
    apply_names_to_graph(g, {id(d): "V2"})
    assert g.warnings == ["V2 has all terminals on one node (short circuit)"]


# ----------------------------------------- net names the drawing invents

@pytest.mark.parametrize("text, want", [
    ("FB", "FB"), ("SW", "SW"), ("GATE", "GATE"), ("OSC", "OSC"),
    ("CTRL", "CTRL"), ("VLOAD", "VLOAD"), ("fb", "FB"),
    ("vcc", "VCC"), ("VOUT", "Vout"),          # the fixed list still wins
])
def test_a_schematics_own_net_names_are_recognised(text, want):
    """The fixed list covered the signals at the *edge* of a drawing. A real
    schematic names its internal nets too, and rejecting those is not
    neutral: the point of writing a name on a wire is to join it to the other
    wire carrying the same name. A boost converter with its feedback marked
    FB at both ends came back with the comparator input on a one-pin net --
    which reads as a wire-tracing failure and is nothing of the kind."""
    from circuit2spice.values import parse_net_label

    assert parse_net_label(text) == want


@pytest.mark.parametrize("text", [
    "R1", "C2", "Rref", "10k", "2N2222", "R1 1k",   # parts, not nets
    "QMOD", "DMOD", "SMOD", "MMOD", "opamp",        # our own model names
    "a much longer phrase", "",
])
def test_a_part_is_never_read_as_a_net_name(text):
    from circuit2spice.values import parse_net_label

    assert parse_net_label(text) is None


# ------------------------- a two-digit designator keeps both its digits

@pytest.mark.parametrize("text, name, value", [
    ("R17 \u03a9", "R17", None),   # value missed, stray unit left behind
    ("R11 \u03a9", "R11", None),
    ("R10 \u03a9", "R10", None),
    ("R12 \u03a9", "R12", None),
    ("R17 100k", "R17", "100k"),   # read properly: unaffected
    ("R12 1k", "R12", "1k"),
    ("R110k", "R1", "10k"),        # run together: the split still applies
    ("R210k", "R2", "10k"),
])
def test_a_space_after_the_index_ends_the_index(text, name, value):
    """Regression. The "R110k means R1 10k" split exists for a *missing*
    space, and applying it across a real one cut ``R17`` into ``R1`` plus the
    value ``7``. Four labels on one drawing produced R1 = 7, R1 = 2, R1 = 1,
    R1 = 0 -- which looks exactly like a loop counter leaking into the
    netlist, and is nothing of the kind.
    """
    from circuit2spice.values import parse_label

    got_name, got_val = parse_label(text)
    assert got_name == name
    assert (got_val.spice if got_val else None) == value


def test_a_bare_unit_is_an_explicable_tail():
    """``R17 Ω`` is R17 with its value unread. Keeping the designator puts the
    part in the "kept the default value" warning instead of losing it."""
    from circuit2spice.values import parse_label

    assert parse_label("R17 \u03a9")[0] == "R17"
    assert parse_label("R17 wibble") == (None, None)


# --------------------------------- a rail marker read as a ground symbol

def test_a_ground_the_drawing_calls_vcc_is_a_rail_marker():
    """A rail marker and a ground symbol differ by one stroke, and the
    detector returned ``gnd`` at 0.58 for the VCC above a 12 V supply. Every
    ground is merged onto one node, so the supply shorted to its own return.
    Nobody labels a ground VCC."""
    from circuit2spice.graph import Detection
    from circuit2spice.values import retag_rail_grounds

    g = Detection("gnd", 85, 472, 106, 489)
    t = Detection("text", 84, 455, 106, 470, text="VCC")
    assert retag_rail_grounds([g, t]) == 1
    assert g.cls == "terminal"


@pytest.mark.parametrize("label", ["GND", "gnd", "", "R1", "10k"])
def test_an_ordinary_ground_is_left_alone(label):
    from circuit2spice.graph import Detection
    from circuit2spice.values import retag_rail_grounds

    g = Detection("gnd", 85, 472, 106, 489)
    t = Detection("text", 84, 455, 106, 470, text=label)
    assert retag_rail_grounds([g, t]) == 0
    assert g.cls == "gnd"


def test_a_distant_label_does_not_retag_a_ground():
    from circuit2spice.graph import Detection
    from circuit2spice.values import retag_rail_grounds

    g = Detection("gnd", 85, 472, 106, 489)
    t = Detection("text", 900, 455, 930, 470, text="VCC")
    assert retag_rail_grounds([g, t]) == 0


@pytest.mark.parametrize("text", ["RL1K", "R110k", "C210u"])
def test_a_run_together_part_label_is_not_a_net_name(text):
    """Regression from widening net names: ``RL`` is not a designator this
    module recognises, so nothing else rejected ``RL1K`` and it became a net.
    A net name does not carry a multiplier."""
    from circuit2spice.values import parse_net_label

    assert parse_net_label(text) is None
