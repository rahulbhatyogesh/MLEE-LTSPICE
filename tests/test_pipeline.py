"""Integration tests: synthetic drawings through wire tracing, and real LTspice.

The LTspice tests are the important ones. A ``.asc`` that merely *looks* right
is worthless -- LTspice fails silently when wires do not meet, opening the
schematic happily with the component floating. So the generated file is handed
back to LTspice and we assert that the circuit it extracts is the circuit we
meant to describe.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from circuit2spice import ltspice
from circuit2spice.asc_writer import snap, to_asc
from circuit2spice.graph import CircuitGraph, Component, Detection
from circuit2spice.topology import build_circuit
from circuit2spice.wires import binarize, build_node_map, remove_thin_strokes

needs_ltspice = pytest.mark.skipif(
    not ltspice.available(), reason="LTspice is not installed"
)


# ------------------------------------------------------------------- fixtures

def _comp(ref, cls, nodes, value, x, y, rot="R0"):
    return Component(ref=ref, cls=cls,
                     det=Detection(cls, x - 30, y - 30, x + 30, y + 30),
                     nodes=nodes, value=value, orientation=rot)


CIRCUITS = {
    "divider": CircuitGraph(
        components=[_comp("V1", "source_voltage", [1, 0], "5", 100, 200),
                    _comp("R1", "resistor", [1, 2], "1k", 300, 120),
                    _comp("R2", "resistor", [2, 0], "2k", 300, 280)],
        n_nodes=3),
    "rc_lowpass": CircuitGraph(
        components=[_comp("V1", "source_voltage", [1, 0], "1", 100, 200),
                    _comp("R1", "resistor", [1, 2], "1k", 300, 120, "R90"),
                    _comp("C1", "capacitor", [2, 0], "1u", 420, 280)],
        n_nodes=3),
    "series_rlc": CircuitGraph(
        components=[_comp("V1", "source_voltage", [1, 0], "1", 100, 300),
                    _comp("R1", "resistor", [1, 2], "10", 250, 100, "R90"),
                    _comp("L1", "inductor", [2, 3], "1m", 420, 100, "R90"),
                    _comp("C1", "capacitor", [3, 0], "1u", 560, 300)],
        n_nodes=4),
    "bjt_stage": CircuitGraph(
        components=[_comp("V1", "source_voltage", [1, 0], "9", 80, 300),
                    _comp("R1", "resistor", [1, 2], "10k", 260, 120),
                    _comp("Q1", "transistor_bjt", [2, 3, 0], "QMOD", 440, 240),
                    _comp("R2", "resistor", [1, 3], "100k", 620, 120)],
        n_nodes=4),
}


# -------------------------------------------------------------- asc structure

def test_snap_is_on_the_sixteen_grid():
    """Off-grid symbols are silently moved by LTspice and stop connecting."""
    for v in (0, 7, 8, 9, 100, 255, -17):
        assert snap(v) % 16 == 0


def test_asc_has_required_header_and_symbols():
    text = to_asc(CIRCUITS["divider"])
    assert text.startswith("Version 4\r\n")
    assert "SHEET 1" in text
    assert text.count("SYMBOL ") == 3
    assert "SYMATTR InstName R1" in text
    assert "SYMATTR Value 2k" in text


def test_every_symbol_origin_is_grid_aligned():
    for name, g in CIRCUITS.items():
        for line in to_asc(g).splitlines():
            if line.startswith("SYMBOL "):
                _, _sym, x, y, _rot = line.split()
                assert int(x) % 16 == 0 and int(y) % 16 == 0, f"{name}: {line}"


def test_ground_becomes_a_flag_not_a_wire():
    text = to_asc(CIRCUITS["divider"])
    assert any(l.startswith("FLAG ") and l.endswith(" 0") for l in text.splitlines())


def test_analysis_directive_is_written():
    assert "!.tran 1m" in to_asc(CIRCUITS["divider"], analysis=".tran 1m")


# --------------------------------------------------------- LTspice round-trips

@needs_ltspice
@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_ltspice_extracts_the_intended_circuit(name, tmp_path):
    from circuit2spice.verify import verify_graph

    res = verify_graph(CIRCUITS[name], tmp_path / f"{name}.asc")
    assert res.ok, f"{name}: {res.summary()}\n{res.netlist_text}"
    assert res.checked_pins > 0


@needs_ltspice
def test_generated_schematic_actually_simulates(tmp_path):
    from circuit2spice.asc_writer import write_asc

    asc = tmp_path / "rlc.asc"
    write_asc(CIRCUITS["series_rlc"], asc, analysis=".ac dec 100 10 1Meg")
    out = ltspice.simulate(asc)
    assert out["raw"] is not None, out["log"][-800:]


@needs_ltspice
def test_divider_netlist_matches_hand_calculation(tmp_path):
    """V1=5, R1=1k, R2=2k in series: LTspice must see exactly that."""
    from circuit2spice.verify import verify_graph

    res = verify_graph(CIRCUITS["divider"], tmp_path / "d.asc")
    els = ltspice.parse_netlist(res.netlist_text)
    assert set(els) == {"V1", "R1", "R2"}
    # R1 and R2 share exactly one net, and it is not ground.
    shared = set(els["R1"]) & set(els["R2"])
    assert len(shared) == 1 and shared.pop() != "0"


# ------------------------------------------------------------- wire extraction

def _draw_loop(w=600, h=400, thickness=5):
    """A rectangular circuit: four wires, two gaps where components sit."""
    img = np.full((h, w, 3), 245, np.uint8)
    p = [(100, 100), (500, 100), (500, 300), (100, 300)]
    for a, b in zip(p, p[1:] + p[:1]):
        cv2.line(img, a, b, (40, 40, 40), thickness)
    return img


def test_components_on_a_loop_find_two_distinct_nodes():
    img = _draw_loop()
    dets = [Detection("resistor", 260, 70, 340, 130),
            Detection("capacitor", 260, 270, 340, 330)]
    g, nm = build_circuit(img, dets, thin_q=None)
    assert len(g.components) == 2
    for c in g.components:
        assert c.is_complete, f"{c.ref} -> {c.nodes}"
        assert len(set(c.nodes)) == 2, f"{c.ref} shorted: {c.nodes}"


def test_a_loop_with_two_components_has_two_nets():
    img = _draw_loop()
    dets = [Detection("resistor", 260, 70, 340, 130),
            Detection("capacitor", 260, 270, 340, 330)]
    g, _ = build_circuit(img, dets, thin_q=None)
    nets = {n for c in g.components for n in c.nodes}
    assert len(nets) == 2


def test_ground_symbol_defines_node_zero():
    img = _draw_loop()
    cv2.line(img, (100, 300), (100, 380), (40, 40, 40), 5)
    dets = [Detection("resistor", 260, 70, 340, 130),
            Detection("capacitor", 260, 270, 340, 330),
            Detection("gnd", 70, 350, 130, 395)]
    g, _ = build_circuit(img, dets, thin_q=None)
    assert g.ground_node == 0
    assert not any("no ground symbol" in w for w in g.warnings)


def test_thin_stroke_removal_deletes_ruling_and_keeps_wires():
    """A thin ruled line crossing a thick wire must not survive."""
    img = np.full((400, 600, 3), 245, np.uint8)
    cv2.line(img, (100, 200), (500, 200), (40, 40, 40), 7)   # pencil wire
    cv2.line(img, (0, 120), (600, 130), (90, 90, 90), 1)     # printed rule
    b = binarize(img)
    cleaned = remove_thin_strokes(b, q=50.0)
    assert cleaned[115:135, 0:40].sum() == 0          # rule line gone at the margin
    assert cleaned[190:210, 280:320].sum() > 0        # wire survives


def test_thin_removal_falls_back_rather_than_erasing_everything():
    """A faintly drawn circuit must not be wiped out by the thickness filter."""
    img = np.full((300, 300, 3), 245, np.uint8)
    cv2.line(img, (50, 150), (250, 150), (60, 60, 60), 3)
    b = binarize(img)
    assert (b > 0).sum() > 0, "precondition: the stroke survives binarisation"
    assert (remove_thin_strokes(b, q=99.0) > 0).sum() > 0


def test_no_detections_yields_an_empty_circuit():
    g, _ = build_circuit(_draw_loop(), [])
    assert g.components == []


# ------------------------------------------------- learned stroke segmentation

def test_unet_method_reports_clearly_when_model_is_absent():
    """Asking for the learned segmenter without a model must say what to do,
    not fail with an obscure torch error."""
    from circuit2spice.segment import StrokeModelMissing, available, find_model

    if available():
        pytest.skip("a stroke model is installed")
    with pytest.raises(StrokeModelMissing) as e:
        find_model()
    assert "train_unet_strokes" in str(e.value)


@pytest.mark.skipif(
    not __import__("circuit2spice.segment", fromlist=["x"]).available(),
    reason="no stroke-segmentation model installed",
)
def test_unet_segmentation_produces_a_usable_mask():
    img = _draw_loop()
    from circuit2spice.segment import segment_strokes

    mask = segment_strokes(img)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    # It must find the drawn rectangle without covering the whole page.
    assert 0.001 < (mask > 0).mean() < 0.5


# ------------------------------------------- branch points erased with a box

def _draw_branch_at_a_terminal():
    """A transistor whose emitter feeds two wires, branching inside its box.

    Left stub -> base; top stub -> collector; from the bottom-right corner the
    emitter runs both right and down. The branch point sits inside the symbol
    box, which is what erasing the box destroys.
    """
    img = np.full((400, 600, 3), 245, np.uint8)
    box = (240, 150, 340, 260)                      # the transistor
    cv2.rectangle(img, (255, 170), (325, 240), (40, 40, 40), 4)   # body ink
    cv2.line(img, (140, 205), (240, 205), (40, 40, 40), 5)        # base, left
    cv2.line(img, (300, 60), (300, 150), (40, 40, 40), 5)         # collector, top
    # emitter: branches at (330, 250), inside the box
    cv2.line(img, (330, 250), (520, 250), (40, 40, 40), 5)
    cv2.line(img, (330, 250), (330, 380), (40, 40, 40), 5)
    return img, box


def test_a_terminal_that_branches_inside_the_box_stays_one_node():
    """Erasing the symbol box must not split one terminal into two nodes."""
    img, box = _draw_branch_at_a_terminal()
    dets = [Detection("transistor_bjt", *box)]
    g, nm = build_circuit(img, dets, thin_q=None)
    q = g.components[0]
    assert q.is_complete, f"Q1 -> {q.nodes}"
    assert len(set(q.nodes)) == 3, f"emitter split or terminals shorted: {q.nodes}"

    # and the two emitter stubs really are one node
    def node_at(x, y, r=6):
        w = nm.labels[y - r : y + r + 1, x - r : x + r + 1]
        hit = w[w > 0]
        assert hit.size, f"no wire near ({x}, {y})"
        return int(hit[0])

    assert node_at(500, 250) == node_at(330, 370), "emitter arrived as two nodes"


def test_branch_repair_never_merges_below_the_port_count():
    """Two genuine terminals must not be fused, however close the box puts them."""
    img = np.full((300, 500, 3), 245, np.uint8)
    cv2.line(img, (60, 150), (200, 150), (40, 40, 40), 5)       # lead 1
    cv2.line(img, (300, 150), (440, 150), (40, 40, 40), 5)      # lead 2
    dets = [Detection("resistor", 200, 110, 300, 190)]
    g, _ = build_circuit(img, dets, thin_q=None)
    r = g.components[0]
    assert len(set(r.nodes)) == 2, f"R1 shorted across its own body: {r.nodes}"


# ---------------------------------------------- a ground symbol with no rail

def test_a_stranded_ground_symbol_is_fused_with_the_return_rail():
    """A gnd symbol on a node of its own is impossible; it means the rail it
    belongs to was drawn without a flag of its own."""
    img = _draw_loop()
    # A source sits in the right-hand wire. Its lower terminal does not join
    # the loop: it ends in a ground symbol, the way a drafter marks a return
    # without drawing the rail. The loop's bottom wire *is* that rail.
    cv2.line(img, (500, 300), (500, 240), (40, 40, 40), 5)      # rail -> source
    img[200:240, 495:506] = 245                                 # gap for the source
    cv2.line(img, (500, 200), (500, 150), (40, 40, 40), 5)      # source -> stub
    img[100:150, 495:506] = 245                                 # severed from the top
    dets = [Detection("resistor", 60, 170, 140, 230),           # on the left wire
            Detection("source_voltage", 470, 195, 530, 245),
            Detection("gnd", 480, 120, 525, 155)]
    g, _ = build_circuit(img, dets, thin_q=None)
    assert g.ground_node == 0
    pins_on_ground = sum(n == 0 for c in g.components for n in c.nodes)
    assert pins_on_ground >= 2, (
        f"ground still stranded: {[(c.ref, c.nodes) for c in g.components]}"
    )
    assert any("stranded" in w or "return rail" in w for w in g.warnings)


# ------------------------------------------- stub collisions in the .asc writer

def _stacked_column():
    """Two 2-pin parts stacked in one column, on four different nets.

    The coordinates are the ones that actually failed: a cap with pins at
    y=432/496 and a resistor with pins at y=560/640, so the facing pins sit
    2*ESCAPE apart. Each escapes 32 units towards the other and they meet at
    y=528 -- one point carrying two net names, and two stubs fused into a
    single wire. The placement is given explicitly so the test pins the
    geometry rather than whatever the auto-layout happens to produce.
    """
    graph = CircuitGraph(
        components=[_comp("C2", "capacitor", [3, 2], "1u", 600, 300),
                    _comp("R5", "resistor", [0, 3], "1k", 600, 420),
                    _comp("V1", "source_voltage", [2, 0], "5", 200, 300),
                    _comp("R1", "resistor", [2, 1], "1k", 200, 420)],
        n_nodes=4)
    placement = {
        "C2": ((976, 432), "R0", [(992, 432), (992, 496)]),
        "R5": ((976, 544), "R0", [(992, 560), (992, 640)]),
        "V1": ((160, 432), "R0", [(176, 448), (176, 528)]),
        "R1": ((160, 640), "R0", [(176, 656), (176, 736)]),
    }
    return graph, placement


def test_no_two_nets_share_a_point_in_the_asc():
    """Every emitted FLAG point belongs to exactly one net."""
    from circuit2spice.asc_writer import to_asc

    graph, placement = _stacked_column()
    text = to_asc(graph, placement=placement)
    seen = {}
    for line in text.splitlines():
        if not line.startswith("FLAG "):
            continue
        _f, x, y, name = line.split()
        prev = seen.setdefault((x, y), name)
        assert prev == name, f"point ({x},{y}) carries both {prev} and {name}"


def test_stubs_of_different_nets_never_overlap():
    """Two stubs meeting end-to-end form one wire, flags or no flags."""
    from circuit2spice.asc_writer import to_asc, _segment_points

    graph, placement = _stacked_column()
    text = to_asc(graph, placement=placement)
    node_of = {}
    for comp in graph.components:
        for node, pin in zip(comp.nodes, placement[comp.ref][2]):
            node_of[pin] = node
    owner = {}
    for line in text.splitlines():
        if not line.startswith("WIRE "):
            continue
        _w, x1, y1, x2, y2 = line.split()
        a, b = (int(x1), int(y1)), (int(x2), int(y2))
        net = node_of.get(a, node_of.get(b))
        for pt in _segment_points(a, b):
            prev = owner.setdefault(pt, net)
            assert prev == net, f"point {pt} is on nets {prev} and {net}"


@needs_ltspice
def test_stacked_components_round_trip(tmp_path):
    """The collision geometry must survive LTspice, not just look right."""
    from circuit2spice.asc_writer import to_asc
    from circuit2spice import ltspice

    graph, placement = _stacked_column()
    asc = tmp_path / "stacked.asc"
    asc.write_text(to_asc(graph, placement=placement), encoding="ascii", newline="")
    els = ltspice.parse_netlist(ltspice.netlist(asc))
    # R5 pin 0 is ground; the collision used to drag it onto C2's lower net.
    assert els["R5"][0] == "0", f"R5 wired to {els['R5']}, expected ground on pin 0"
    # C2 pin 0 and R5 pin 1 share node 3 by construction; the *facing* pins
    # (C2 pin 1, R5 pin 0) are different nets and must stay apart.
    assert els["C2"][0] == els["R5"][1], (els["C2"], els["R5"])
    assert els["C2"][1] != els["R5"][0], (els["C2"], els["R5"])


# ------------------------------------------------------- reading .raw results

@needs_ltspice
def test_operating_point_is_read_back_with_the_right_word_sizes(tmp_path):
    """A .raw real plot stores variable 0 as float64 and the rest as float32.

    Reading them all as float64 does not raise -- it returns numbers like
    8.9e245 -- so this asserts the values are physically sane.
    """
    from circuit2spice import ltspice
    from circuit2spice.asc_writer import write_asc

    asc = tmp_path / "divider.asc"
    write_asc(CIRCUITS["divider"], asc, analysis=".op")
    sim = ltspice.simulate(asc)
    assert sim["raw"] is not None, sim["log"][-800:]

    op = ltspice.operating_point(sim["raw"])
    volts = {k: v for k, v in op.items() if k.lower().startswith("v(")}
    assert volts, op
    assert all(abs(v) < 1e3 for v in volts.values()), volts
    # V1=5 across R1=1k + R2=2k, so the divider tap must sit at 10/3 V.
    assert any(abs(v - 10 / 3) < 1e-3 for v in volts.values()), volts


def test_verify_does_not_clobber_the_analysis_directive(tmp_path):
    """verify_graph rewrites the .asc, so it must keep the caller's analysis.

    Without this, --verify silently reverted --analysis to the default and the
    simulation produced an operating point when a transient was asked for.
    """
    from circuit2spice.asc_writer import write_asc

    asc = tmp_path / "rlc.asc"
    write_asc(CIRCUITS["series_rlc"], asc, analysis=".tran 0 5m 0 2u")
    assert "!.tran 0 5m 0 2u" in asc.read_text()

    # Re-writing through the verify path must preserve it.
    write_asc(CIRCUITS["series_rlc"], asc, analysis=".tran 0 5m 0 2u")
    assert "!.tran 0 5m 0 2u" in asc.read_text()
    assert "!.op" not in asc.read_text()


@needs_ltspice
def test_pipeline_verify_keeps_a_transient_analysis(tmp_path):
    from circuit2spice.verify import verify_graph

    asc = tmp_path / "t.asc"
    res = verify_graph(CIRCUITS["series_rlc"], asc, analysis=".tran 0 5m 0 2u")
    assert res.ok, res.summary()
    assert "!.tran 0 5m 0 2u" in asc.read_text(), asc.read_text()[-300:]


@needs_ltspice
def test_transient_time_axis_is_monotonic(tmp_path):
    """LTspice sets the sign bit on ~half of a transient's timestamps.

    Left as-is the time column alternates sign, the axis runs negative and any
    plot drawn from it is scrambled -- a parsing fault that looks like a
    simulation fault.
    """
    from circuit2spice import ltspice
    from circuit2spice.asc_writer import write_asc

    # A DC source settles and LTspice compresses the run to two points, so the
    # sign-bit pattern never appears. Drive it with a sine instead.
    driven = CircuitGraph(
        components=[_comp("V1", "source_voltage", [1, 0], "SINE(0 1 5k)", 100, 300),
                    _comp("R1", "resistor", [1, 2], "10", 250, 100, "R90"),
                    _comp("L1", "inductor", [2, 3], "1m", 420, 100, "R90"),
                    _comp("C1", "capacitor", [3, 0], "1u", 560, 300)],
        n_nodes=4)
    asc = tmp_path / "rlc.asc"
    write_asc(driven, asc, analysis=".tran 0 1m 0 2u")
    sim = ltspice.simulate(asc)
    assert sim["raw"] is not None, sim["log"][-800:]

    r = ltspice.read_raw(sim["raw"])
    assert r["plotname"].lower().startswith("transient"), r["plotname"]
    t = r["data"]["time"]
    assert len(t) > 10, len(t)
    assert all(v >= 0 for v in t), "negative timestamps survived"
    assert all(a <= b for a, b in zip(t, t[1:])), "time axis is not monotonic"
    assert abs(max(t) - 1e-3) < 1e-4, max(t)


def test_a_part_can_be_reversed_by_hand():
    """Polarity read from the drawing can still be wrong, so it must be fixable.

    Diode orientation is read by ``diode_net`` now, but it abstains when the
    agreement gate is not met, and source polarity is not read at all, so
    `--flip D1` remains the exact override.
    """
    from types import SimpleNamespace

    graph = SimpleNamespace(
        components=[SimpleNamespace(ref="D1", nodes=[3, 7]),
                    SimpleNamespace(ref="R1", nodes=[1, 2])],
        warnings=[])
    for ref in ["d1"]:                       # case must not matter
        for c in graph.components:
            if c.ref.upper() == ref.upper():
                c.nodes = list(reversed(c.nodes))
                graph.warnings.append(f"reversed {c.ref} by hand")
    assert graph.components[0].nodes == [7, 3]
    assert graph.components[1].nodes == [1, 2]
    assert graph.warnings == ["reversed D1 by hand"]
