"""Correcting the pipeline by hand, and the net-label fusion bug.

Three of these overrides exist because some things are not learnable from the
only dataset there is. CGHD holds **1314 voltage sources and no current
sources at all**, and 977 FETs with no orientation annotation, so a detector
trained on it cannot be taught either one. Where measurement is impossible,
an explicit lever is the honest answer -- but it has to be a lever the user
can actually reach, which is most of what these tests pin.
"""

from __future__ import annotations

import pytest

from circuit2spice.graph import CircuitGraph, Component, Detection
from circuit2spice.topology import add_supplies, reclassify, set_pins
from circuit2spice.values import name_nets_from_labels


def _comp(ref, cls, nodes, value, x=0.0, y=0.0):
    return Component(ref=ref, cls=cls, det=Detection(cls, x, y, x + 20, y + 20),
                     nodes=list(nodes), value=value)


def _graph():
    return CircuitGraph(
        components=[_comp("V1", "source_voltage", [1, 0], "5", 0, 0),
                    _comp("V2", "source_voltage", [2, 3], "5", 100, 0),
                    _comp("M1", "transistor_fet", [0, 1, 2], "MMOD", 200, 0),
                    _comp("R1", "resistor", [3, 0], "1k", 300, 0)],
        n_nodes=4, ground_node=0, image_size=(400, 100),
        net_names={3: "VCC"})


# ------------------------------------------------------------ re-typing

def test_a_voltage_source_can_be_corrected_to_a_current_source():
    """The case that forced this: CGHD has no current sources to learn from,
    so the detector calls one a voltage source at 0.91 and always will."""
    g = _graph()
    reclassify(g, {"V2": "source_current"})
    c = next(c for c in g.components if c.cls == "source_current")
    assert c.ref == "I1"
    assert c.nodes == [2, 3], "the wiring came from the ink and must not move"
    assert any("re-typed V2" in w for w in g.warnings)


def test_a_default_value_is_replaced_but_a_read_one_is_kept():
    g = _graph()
    g.components[1].value = "5"                   # still source_voltage's default
    reclassify(g, {"V2": "source_current"})
    assert next(c for c in g.components if c.ref == "I1").value == "1m"

    g = _graph()
    g.components[1].value = "12"                  # read off the drawing
    reclassify(g, {"V2": "source_current"})
    assert next(c for c in g.components if c.ref == "I1").value == "12"


def test_a_class_with_a_different_port_count_is_refused():
    """Three terminals were never looked for, so they cannot be invented."""
    g = _graph()
    reclassify(g, {"V2": "transistor_bjt"})
    assert g.components[1].cls == "source_voltage"
    assert any("cannot re-type V2" in w for w in g.warnings)


def test_an_unknown_class_or_ref_is_reported_not_ignored():
    g = _graph()
    reclassify(g, {"V2": "flux_capacitor", "Z9": "resistor"})
    assert any("unknown class" in w for w in g.warnings)
    assert any("no component called Z9" in w for w in g.warnings)


def test_the_new_ref_never_collides():
    g = _graph()
    g.components.append(_comp("I1", "source_current", [1, 2], "1m", 400, 0))
    reclassify(g, {"V2": "source_current"})
    refs = [c.ref for c in g.components]
    assert len(refs) == len(set(refs))
    assert "I2" in refs


# ------------------------------------------------------------- supplies

def test_a_rail_can_be_driven_by_name():
    g = _graph()
    add_supplies(g, {"VCC": "5"})
    v = g.components[-1]
    assert v.cls == "source_voltage" and v.value == "5"
    assert v.nodes == [3, 0]
    assert v.ref not in [c.ref for c in g.components[:-1]]


def test_adding_a_supply_says_the_drawing_did_not_ask_for_it():
    """It is an invention, and an invention that is not announced is a lie
    about what was read off the page."""
    g = _graph()
    add_supplies(g, {"VCC": "5"})
    assert any("the drawing does not say this" in w for w in g.warnings)


def test_a_supply_can_name_a_node_by_number():
    g = _graph()
    add_supplies(g, {"2": "3.3"})
    assert g.components[-1].nodes == [2, 0]


def test_a_supply_across_ground_is_refused():
    g = _graph()
    add_supplies(g, {"0": "5"})
    assert len(g.components) == 4
    assert any("across ground" in w for w in g.warnings)


def test_an_unknown_net_is_reported():
    g = _graph()
    add_supplies(g, {"VDD": "5"})
    assert len(g.components) == 4
    assert any("no net called" in w for w in g.warnings)


# ----------------------------------------------------------- pin orders

def test_pins_can_express_a_permutation_flip_cannot():
    """A FET with drain and gate transposed is the motivating case: reversing
    a three-pin part swaps the outer two and can never move the middle one."""
    g = _graph()
    set_pins(g, {"M1": "N001,N002,0"})
    assert next(c for c in g.components if c.ref == "M1").nodes == [1, 2, 0]


def test_pins_accept_a_net_name_and_ground():
    g = _graph()
    set_pins(g, {"R1": "VCC,0"})
    assert next(c for c in g.components if c.ref == "R1").nodes == [3, 0]


def test_the_wrong_number_of_pins_is_refused():
    g = _graph()
    before = list(g.components[2].nodes)
    set_pins(g, {"M1": "N001,0"})
    assert g.components[2].nodes == before
    assert any("takes 3 nodes, got 2" in w for w in g.warnings)


def test_an_unknown_net_in_a_pin_list_is_refused_whole():
    g = _graph()
    before = list(g.components[2].nodes)
    set_pins(g, {"M1": "N001,Nowhere,0"})
    assert g.components[2].nodes == before
    assert any("no net called 'Nowhere'" in w for w in g.warnings)


# ------------------------------------------- two labels, one net (the bug)

def _labelled(name_a="VCC", name_b="VCC"):
    """Two terminals on two different nets, each with a label beside it."""
    ta = Detection("terminal", 0, 20, 20, 40)
    tb = Detection("terminal", 100, 20, 120, 40)
    la = Detection("text", 0, 0, 20, 18, text=name_a)
    lb = Detection("text", 100, 0, 120, 18, text=name_b)
    g = CircuitGraph(
        components=[_comp("R1", "resistor", [1, 0], "1k"),
                    _comp("R2", "resistor", [2, 0], "1k")],
        n_nodes=2, ground_node=0, image_size=(200, 100),
        terminal_nodes={id(ta): 1, id(tb): 2})
    return g, [ta, tb, la, lb]


def test_two_terminals_labelled_the_same_are_one_net():
    """The bug: a drafter writes VCC at the top of two rails *because* that
    joins them. The second label used to be skipped as a duplicate name, so
    the circuit stayed cut in half at the one place the drawing was explicit.
    """
    g, dets = _labelled("VCC", "VCC")
    name_nets_from_labels(g, dets)
    nets = {c.nodes[0] for c in g.components}
    assert len(nets) == 1, "both rails must land on one node"
    assert g.node_name(nets.pop()) == "VCC"


def test_two_different_names_stay_two_nets():
    g, dets = _labelled("Vout", "VCC")
    name_nets_from_labels(g, dets)
    assert {g.node_name(c.nodes[0]) for c in g.components} == {"Vout", "VCC"}


def test_a_net_labelled_gnd_is_merged_into_ground():
    """SPICE node 0 *is* ground; a second net called GND is a contradiction."""
    g, dets = _labelled("GND", "VCC")
    name_nets_from_labels(g, dets)
    assert g.components[0].nodes[0] == g.ground_node
    assert g.node_name(g.components[0].nodes[0]) == "0"


def test_merging_keeps_the_terminal_map_usable():
    g, dets = _labelled("VCC", "VCC")
    name_nets_from_labels(g, dets)
    assert set(g.terminal_nodes.values()) == {g.components[0].nodes[0]}


def test_merge_nodes_is_a_no_op_on_itself():
    g = _graph()
    before = [list(c.nodes) for c in g.components]
    g.merge_nodes(2, 2)
    assert [c.nodes for c in g.components] == before


@pytest.mark.parametrize("name", ["vcc", "Vcc", "VCC"])
def test_the_label_is_matched_whatever_its_case(name):
    g, dets = _labelled("VCC", name)
    name_nets_from_labels(g, dets)
    assert len({c.nodes[0] for c in g.components}) == 1


# ------------------------------------------- the switch's control pins

def _switched():
    g = CircuitGraph(
        components=[_comp("S1", "switch", [1, 2], "SMOD"),
                    _comp("X1", "opamp", [1, 2, 3], "opamp", 50, 0)],
        n_nodes=3, ground_node=0, image_size=(100, 100))
    return g


def test_a_switch_defaults_to_control_pins_on_ground():
    """Which leaves it open -- the honest default when the drawing is silent,
    but it has to be the *same* default in the netlist and the schematic."""
    g = _switched()
    assert g.wired_nodes(g.components[0]) == [1, 2, 0, 0]


def test_a_switch_control_can_be_driven():
    g = _switched()
    set_pins(g, {"S1": "N001,N002,N003,0"})
    s = g.components[0]
    assert s.nodes == [1, 2] and s.control == [3, 0]
    assert g.wired_nodes(s) == [1, 2, 3, 0]


def test_the_netlist_writes_the_control_pins():
    from circuit2spice.netlist import to_netlist

    g = _switched()
    set_pins(g, {"S1": "N001,N002,N003,0"})
    line = next(ln for ln in to_netlist(g).splitlines() if ln.startswith("S1"))
    assert line.split()[1:5] == ["N001", "N002", "N003", "0"]


def test_a_switch_still_accepts_just_its_two_terminals():
    g = _switched()
    set_pins(g, {"S1": "N002,N001"})
    assert g.components[0].nodes == [2, 1]
    assert g.components[0].control is None


def test_three_nodes_on_a_switch_is_refused():
    g = _switched()
    set_pins(g, {"S1": "N001,N002,N003"})
    assert g.components[0].nodes == [1, 2]
    assert any("or 4, with the control pair" in w for w in g.warnings)


def test_merging_a_net_follows_the_control_pins_too():
    g = _switched()
    set_pins(g, {"S1": "N001,N002,N003,0"})
    g.merge_nodes(1, 3)
    assert g.components[0].control == [1, 0]


# ------------------------------------- re-typing reaches the detection

def test_re_typing_corrects_the_detection_as_well():
    """Label matching keys off the *detection's* class, so leaving it stale
    means a re-typed inductor can never take "L1" or "100u" off the page."""
    g = _graph()
    reclassify(g, {"R1": "inductor"})
    c = next(c for c in g.components if c.cls == "inductor")
    assert c.det.cls == "inductor"
