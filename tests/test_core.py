"""Unit tests for the pure-logic parts: taxonomy, symbols, netlist, comparison."""

from __future__ import annotations

import pytest

from circuit2spice.classes import (ERASE_BEFORE_WIRES, NAMES, PLACEABLE,
                                   merge_label, spec)
from circuit2spice.graph import CircuitGraph, Component, Detection, UnionFind
from circuit2spice.netcompare import compare, is_isomorphic, netlist_to_graph
from circuit2spice.netlist import to_netlist
from circuit2spice.symbols import load_symbol, rotate


# --------------------------------------------------------------------- classes

@pytest.mark.parametrize("raw,expected", [
    ("resistor", "resistor"),
    ("resistor.adjustable", "resistor"),
    ("resistor.photo", "resistor"),
    ("capacitor.unpolarized", "capacitor"),
    ("capacitor.polarized", "capacitor"),
    ("capacitor.adjustable", "capacitor"),
    ("diode.light_emitting", "diode"),
    ("diode.zener", "diode"),
    ("diode.thyrector", "diode"),
    ("transistor.bjt", "transistor_bjt"),
    ("transistor.fet", "transistor_fet"),
    ("voltage.dc", "source_voltage"),
    ("voltage.ac", "source_voltage"),
    ("voltage.battery", "source_voltage"),
    ("operational_amplifier", "opamp"),
    ("operational_amplifier.schmitt_trigger", "opamp"),
    ("gnd", "gnd"),
    ("junction", "junction"),
    ("crossover", "crossover"),
    ("text", "text"),
])
def test_merge_known_labels(raw, expected):
    assert merge_label(raw) == expected


@pytest.mark.parametrize("raw", ["integrated_circuit.ne555", "vss", "and",
                                 "explanatory", "relay", "motor", "unknown"])
def test_unsupported_labels_become_catch_all(raw):
    """Must not be dropped: a symbol we cannot see is one we cannot erase."""
    assert merge_label(raw) == "other_component"


def test_merge_is_case_and_separator_insensitive():
    assert merge_label("  Resistor.Adjustable ") == "resistor"
    assert merge_label("transistor-bjt") == "transistor_bjt"


def test_merge_handles_empty():
    assert merge_label(None) is None
    assert merge_label("") is None


def test_every_merged_name_has_a_spec():
    for n in NAMES:
        assert spec(n).name == n


def test_placeable_components_have_symbols_and_values():
    for n in PLACEABLE:
        s = spec(n)
        assert s.spice_prefix and s.ltspice_symbol and s.default_value


def test_catch_all_is_erased_but_not_placed():
    assert "other_component" in ERASE_BEFORE_WIRES
    assert "other_component" not in PLACEABLE


def test_junctions_are_not_erased():
    """A junction dot carries connectivity; erasing it would cut the net."""
    assert "junction" not in ERASE_BEFORE_WIRES


# --------------------------------------------------------------------- symbols

def test_rotation_is_the_ltspice_convention():
    # Verified against LTspice 26.1.1 by netlisting a rotated resistor.
    assert rotate(16, 96, "R0") == (16, 96)
    assert rotate(16, 96, "R90") == (-96, 16)
    assert rotate(16, 96, "R180") == (-16, -96)
    assert rotate(16, 96, "R270") == (96, -16)


def test_rotation_four_times_is_identity():
    x, y = 16, 96
    pt = (x, y)
    for _ in range(4):
        pt = rotate(pt[0], pt[1], "R90")
    assert pt == (x, y)


def test_rotation_rejects_nonsense():
    with pytest.raises(ValueError):
        rotate(0, 0, "R45")


def test_resistor_pin_geometry():
    """Pin B is at y=96, past the end of the drawn body -- the classic trap."""
    res = load_symbol("res")
    assert len(res.pins) == 2
    assert (res.pins[0].x, res.pins[0].y) == (16, 16)
    assert (res.pins[1].x, res.pins[1].y) == (16, 96)


def test_pin_positions_translate():
    res = load_symbol("res")
    assert res.pin_positions((288, 80), "R0") == [(304, 96), (304, 176)]


def test_three_terminal_symbols_have_three_pins():
    for name in ("npn", "nmos"):
        assert len(load_symbol(name).pins) == 3


# --------------------------------------------------------------------- netlist

def _divider() -> CircuitGraph:
    def c(ref, cls, nodes, value, x, y):
        return Component(ref=ref, cls=cls,
                         det=Detection(cls, x - 10, y - 10, x + 10, y + 10),
                         nodes=nodes, value=value)
    return CircuitGraph(
        components=[c("V1", "source_voltage", [1, 0], "5", 0, 0),
                    c("R1", "resistor", [1, 2], "1k", 50, 0),
                    c("R2", "resistor", [2, 0], "2k", 50, 50)],
        n_nodes=3)


def test_netlist_text_is_well_formed():
    text = to_netlist(_divider(), title="divider")
    assert "R1 N001 N002 1k" in text
    assert "R2 N002 0 2k" in text
    assert "V1 N001 0 DC 5" in text
    assert text.strip().endswith(".end")


def test_ground_is_always_node_zero():
    g = _divider()
    assert g.node_name(g.ground_node) == "0"
    assert g.node_name(1) == "N001"


def test_model_cards_emitted_only_when_needed():
    g = _divider()
    assert ".model" not in to_netlist(g)
    g.components[1] = Component("D1", "diode", g.components[1].det, [1, 2], "DMOD")
    assert ".model DMOD" in to_netlist(g)


# ------------------------------------------------------------------ comparison

def test_identical_netlists_are_isomorphic():
    a = {"R1": ["N001", "N002"], "R2": ["N002", "0"], "V1": ["N001", "0"]}
    assert is_isomorphic(netlist_to_graph(a), netlist_to_graph(dict(a)))


def test_renaming_nets_does_not_break_isomorphism():
    a = {"R1": ["N001", "N002"], "R2": ["N002", "0"], "V1": ["N001", "0"]}
    b = {"R1": ["A", "B"], "R2": ["B", "0"], "V1": ["A", "0"]}
    assert is_isomorphic(netlist_to_graph(a), netlist_to_graph(b))


def test_different_topology_is_not_isomorphic():
    series = {"R1": ["N001", "N002"], "R2": ["N002", "0"], "V1": ["N001", "0"]}
    parallel = {"R1": ["N001", "0"], "R2": ["N001", "0"], "V1": ["N001", "0"]}
    assert not is_isomorphic(netlist_to_graph(series), netlist_to_graph(parallel))


def test_ground_pinning_can_be_disabled():
    """Same cycle, different choice of reference node."""
    a = {"R1": ["0", "N001"], "V1": ["N001", "N002"], "R2": ["N002", "0"]}
    b = {"V1": ["N001", "0"], "R1": ["N001", "N002"], "R2": ["N002", "0"]}
    assert not is_isomorphic(netlist_to_graph(a, True), netlist_to_graph(b, True))
    assert is_isomorphic(netlist_to_graph(a, False), netlist_to_graph(b, False))


def test_compare_reports_perfect_scores_for_a_match():
    a = {"R1": ["N001", "N002"], "R2": ["N002", "0"], "V1": ["N001", "0"]}
    r = compare("x", "C1", dict(a), a)
    assert r.exact and r.bom_f1 == 1.0 and r.sig_f1 == 1.0


def test_unconnected_ports_do_not_silently_join():
    """Two unresolved ports must not be treated as one net just because both
    failed; each gets a unique NC name upstream."""
    ref = {"R1": ["N001", "N002"], "R2": ["N002", "0"], "V1": ["N001", "0"]}
    joined = {"R1": ["N001", "NC"], "R2": ["NC", "0"], "V1": ["N001", "0"]}
    unique = {"R1": ["N001", "NC_1"], "R2": ["NC_2", "0"], "V1": ["N001", "0"]}
    assert compare("x", "C1", joined, ref).exact        # NC collapsed -> matches
    assert not compare("x", "C1", unique, ref).exact    # kept apart -> does not


# ------------------------------------------------------------------- unionfind

def test_union_find_groups_transitively():
    uf = UnionFind()
    uf.union(1, 2)
    uf.union(2, 3)
    uf.find(4)
    assert uf.find(1) == uf.find(3)
    assert uf.find(4) != uf.find(1)
    assert sorted(len(v) for v in uf.groups().values()) == [1, 3]
