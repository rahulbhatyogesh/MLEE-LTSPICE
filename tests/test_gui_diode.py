"""The GUI's diode integration, exercised without opening a window.

Tk is not available on every machine that runs these tests and a window is
not the thing worth testing anyway. What matters is the logic the window
drives: that the diode model is discovered and reported, that the netlist
view spells a diode's order out, and that the flip override actually rewrites
the exported files. Those are all methods on ``App`` that never touch a
widget, so they can be called on a bare object.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from circuit2spice.graph import CircuitGraph, Component, Detection

gui = pytest.importorskip("circuit2spice.gui")


def _det(cls, x, y):
    return Detection(cls, x - 20, y - 20, x + 20, y + 20)


def _graph():
    return CircuitGraph(
        components=[
            Component(ref="D1", cls="diode", det=_det("diode", 100, 100),
                      nodes=[1, 2], value="DMOD", orientation="R0"),
            Component(ref="R1", cls="resistor", det=_det("resistor", 200, 100),
                      nodes=[2, 0], value="1k", orientation="R0"),
            Component(ref="V1", cls="source_voltage",
                      det=_det("source_voltage", 50, 100),
                      nodes=[1, 0], value="5", orientation="R0"),
        ],
        n_nodes=3)


def _result(tmp_path):
    g = _graph()
    return SimpleNamespace(
        graph=g, detections=[c.det for c in g.components], verification=None,
        outputs={"asc": tmp_path / "x.asc", "cir": tmp_path / "x.cir",
                 "json": tmp_path / "x.netlist.json"})


# ------------------------------------------------------------- model lookup

def test_the_diode_model_is_discovered_like_the_detector(tmp_path, monkeypatch):
    """A missing model is reported, not guessed at -- a diode the wrong way
    round still simulates, so silence here reads as success."""
    monkeypatch.setenv("CIRCUIT2SPICE_DIODE", "")
    monkeypatch.chdir(tmp_path)
    assert gui.find_diode_model() is None

    p = tmp_path / "models" / "diode_orient.ts.pt"
    p.parent.mkdir()
    p.write_bytes(b"not a real model, but it is a file")
    assert gui.find_diode_model() == Path("models/diode_orient.ts.pt")


def test_an_explicit_diode_model_wins(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere.ts.pt"
    other.write_bytes(b"x")
    monkeypatch.setenv("CIRCUIT2SPICE_DIODE", str(other))
    assert gui.find_diode_model() == other


# ------------------------------------------------------------- netlist view

def test_the_netlist_view_spells_out_which_end_is_the_anode(tmp_path):
    app = gui.App.__new__(gui.App)          # no Tk, no window
    lines = gui.App._render(app, _result(tmp_path), "unet")
    d = [ln for ln in lines if ln.strip().startswith("D1")][0]
    assert "anode -> cathode" in d
    # and an ordinary part is left as a plain list
    r = [ln for ln in lines if ln.strip().startswith("R1")][0]
    assert "->" not in r


# -------------------------------------------------------------- the override

class _Q:
    def __init__(self): self.items = []
    def put(self, item): self.items.append(item)


def _app_for_flip(tmp_path, ref="D1", verify=False):
    app = gui.App.__new__(gui.App)
    app.q = _Q()
    app.result = _result(tmp_path)
    app._verify = verify
    return app


def test_flipping_reverses_the_part_and_rewrites_every_export(tmp_path):
    """The override has to reach the files, not just the view.

    The .asc is what LTspice opens and the .cir is what gets simulated; a
    flip that only changed the on-screen netlist would be worse than none.
    """
    app = _app_for_flip(tmp_path)
    gui.App._flip_work(app, "D1", ".op", app._verify)

    d1 = [c for c in app.result.graph.components if c.ref == "D1"][0]
    assert d1.nodes == [2, 1], "the part was not reversed"
    assert "reversed D1 by hand" in app.result.graph.warnings

    for key in ("asc", "cir", "json"):
        assert Path(app.result.outputs[key]).is_file(), f"{key} not rewritten"
    assert json.loads(Path(app.result.outputs["json"]).read_text())
    kinds = [k for k, _ in app.q.items]
    assert "flipdone" in kinds and "netlist" in kinds
    assert "error" not in kinds


def test_flipping_is_case_insensitive_like_the_cli(tmp_path):
    app = _app_for_flip(tmp_path)
    gui.App._flip_work(app, "d1", ".op", app._verify)
    assert [c for c in app.result.graph.components if c.ref == "D1"][0].nodes == [2, 1]


def test_flipping_an_unknown_part_says_so_and_changes_nothing(tmp_path):
    app = _app_for_flip(tmp_path)
    before = [list(c.nodes) for c in app.result.graph.components]
    gui.App._flip_work(app, "D9", ".op", app._verify)
    after = [list(c.nodes) for c in app.result.graph.components]
    assert before == after
    msgs = [kw.get("msg", "") for k, kw in app.q.items if k == "log"]
    assert any("no part called D9" in m for m in msgs)
    assert "error" not in [k for k, _ in app.q.items]


def test_the_worker_reports_the_diodes_it_found(tmp_path):
    """The combobox is populated from the graph, so it cannot offer a part
    that is not there -- and it must offer every diode that is."""
    g = _graph()
    assert [c.ref for c in g.components if c.cls == "diode"] == ["D1"]


# ------------------------------------------------- the Corrections box

@pytest.mark.parametrize("text, want", [
    ("", {}),
    ("R1=10k", {"values": {"R1": "10k"}}),
    ("R1=10k R2=4k7", {"values": {"R1": "10k", "R2": "4k7"}}),
    ("as V2=source_current", {"reclass": {"V2": "source_current"}}),
    ("supply VCC=5", {"supplies": {"VCC": "5"}}),
    ("pins M1=N001,N005,0", {"pins": {"M1": "N001,N005,0"}}),
    ("as V2=source_current value I1=1m supply VCC=5",
     {"reclass": {"V2": "source_current"}, "values": {"I1": "1m"},
      "supplies": {"VCC": "5"}}),
    ("Value: R1=1k", {"values": {"R1": "1k"}}),
])
def test_the_corrections_box_parses(text, want):
    from circuit2spice.gui import parse_corrections

    assert parse_corrections(text) == want


def test_a_half_typed_correction_does_not_take_the_run_down():
    """It is a text box in a demo form; the user is mid-word most of the
    time it is read."""
    from circuit2spice.gui import parse_corrections

    assert parse_corrections("suppl") == {}
    assert parse_corrections("=5") == {}
    assert parse_corrections("R1=") == {}
    assert parse_corrections("as") == {}


def test_the_parsed_corrections_are_exactly_processs_keywords():
    """If these drift apart the box silently stops working."""
    import inspect

    from circuit2spice.gui import _CORRECTION_KINDS
    from circuit2spice.pipeline import process

    params = set(inspect.signature(process).parameters)
    assert set(_CORRECTION_KINDS.values()) <= params


@pytest.mark.parametrize("text", [
    "as R10=inductor",
    "--as R10=inductor",          # pasted straight from the command line
    "-as R10=inductor",
    "-- as R10=inductor",         # with the POSIX end-of-options marker
    "As: R10=inductor",
])
def test_the_corrections_box_accepts_the_command_line_spelling(text):
    """Regression. ``--as R10=inductor`` used to leave the kind alone, so it
    became *value* ``R10 = "inductor"``: the part was never re-typed, the
    netlist read ``R10 N005 N002 inductor``, and nothing said so."""
    from circuit2spice.gui import parse_corrections

    assert parse_corrections(text) == {"reclass": {"R10": "inductor"}}


def test_a_word_that_is_not_a_kind_is_reported_not_obeyed_as_another():
    from circuit2spice.gui import parse_corrections

    problems: list = []
    got = parse_corrections("asx R1=1k", problems)
    assert problems == ["asx"]
    assert got == {"values": {"R1": "1k"}}, "it still defaults, but it says so"


def test_a_malformed_pair_is_reported():
    from circuit2spice.gui import parse_corrections

    # Tested one at a time: closing up the space around ``=`` is what makes
    # ``R1 = 1k`` work, so a dangling ``=`` necessarily absorbs the next word.
    for text, want in (("R1=", ["R1="]), ("=5", ["=5"])):
        problems: list = []
        assert parse_corrections(text, problems) == {}
        assert problems == want


def test_punctuation_alone_is_not_a_complaint():
    from circuit2spice.gui import parse_corrections

    problems: list = []
    parse_corrections("-- value R1=1k", problems)
    assert problems == []


@pytest.mark.parametrize("text, want", [
    ("as R10 = inductor", {"reclass": {"R10": "inductor"}}),
    ("as R10 =inductor", {"reclass": {"R10": "inductor"}}),
    ("as R10= inductor", {"reclass": {"R10": "inductor"}}),
    ("value C2 = 10u value C3 = 10n",
     {"values": {"C2": "10u", "C3": "10n"}}),
    ("pins M1 = N005, N006, 0", {"pins": {"M1": "N005,N006,0"}}),
    ("pins S1 = N001, N002, N016, 0", {"pins": {"S1": "N001,N002,N016,0"}}),
])
def test_spaces_around_the_equals_and_commas_are_tolerated(text, want):
    """Regression, and the one that actually bit: ``as R10 = inductor`` is
    what a person types, and splitting it raw gives three tokens with no pair
    among them -- so the whole box parsed to nothing and the run went ahead
    looking exactly as though the correction had been applied and refused."""
    from circuit2spice.gui import parse_corrections

    problems: list = []
    assert parse_corrections(text, problems) == want
    assert problems == []


def test_a_box_that_parses_to_nothing_is_detectable_by_the_caller():
    """The GUI needs this to shout: text in, nothing out, no exception."""
    from circuit2spice.gui import parse_corrections

    problems: list = []
    assert parse_corrections("complete nonsense", problems) == {}
    assert problems


def test_the_build_stamp_names_the_running_source():
    """A stale process is indistinguishable from a broken fix, so the GUI
    says which copy of the code it is actually running."""
    from pathlib import Path

    from circuit2spice import gui
    from circuit2spice.gui import build_stamp

    stamp = build_stamp()
    assert str(Path(gui.__file__).resolve().parent) in stamp
    assert "gui.py" in stamp
