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
