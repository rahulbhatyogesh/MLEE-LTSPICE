"""The Ollama GPU-offload check.

A reader split between GPU and CPU returns exactly the right characters and
takes about ten times as long. That is the worst shape a regression can have:
nothing looks broken, so the slowdown gets blamed on whatever changed most
recently. These tests pin the detection, the threshold and the silence when
there is nothing to say.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from circuit2spice import ocr


def _ps(models):
    """A fake /api/ps body."""
    return json.dumps({"models": models}).encode()


class _Resp:
    def __init__(self, body): self._b = body
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch_urlopen(monkeypatch, body, expect="/api/ps"):
    def fake(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        assert expect in url, url
        if body is None:
            raise OSError("no service")
        return _Resp(body)
    monkeypatch.setattr("urllib.request.urlopen", fake)


# ----------------------------------------------------------------- the probe

def test_a_fully_resident_model_reports_one(monkeypatch):
    _patch_urlopen(monkeypatch, _ps([
        {"name": "qwen3-vl:2b-instruct-q8_0", "size": 3_000_000_000,
         "size_vram": 3_000_000_000}]))
    frac, vram, total = ocr.ollama_offload("qwen3-vl:2b-instruct-q8_0")
    assert frac == pytest.approx(1.0)
    assert (vram, total) == (3_000_000_000, 3_000_000_000)


def test_a_split_model_reports_the_fraction(monkeypatch):
    """The real numbers seen when training had the card: 1.2 of 3.1 GB."""
    _patch_urlopen(monkeypatch, _ps([
        {"name": "qwen3-vl:2b-instruct-q8_0", "size": 3_114_438_491,
         "size_vram": 1_204_279_050}]))
    frac, _, _ = ocr.ollama_offload("qwen3-vl:2b-instruct-q8_0")
    assert 0.38 == pytest.approx(frac, abs=0.01)


def test_a_model_that_is_not_loaded_is_not_a_verdict(monkeypatch):
    _patch_urlopen(monkeypatch, _ps([]))
    assert ocr.ollama_offload("qwen3-vl:2b-instruct-q8_0") is None


def test_no_service_is_not_a_verdict(monkeypatch):
    """Absence of Ollama is the normal case and must cost nothing and say
    nothing -- the same contract ``ollama_models`` already keeps."""
    _patch_urlopen(monkeypatch, None)
    assert ocr.ollama_offload("whatever") is None


def test_a_tag_prefix_still_matches(monkeypatch):
    """`/api/ps` may name a build differently from the tag that was asked
    for, exactly as `/api/tags` does."""
    _patch_urlopen(monkeypatch, _ps([
        {"name": "qwen3-vl:2b-instruct-q8_0-llamacpp", "size": 100,
         "size_vram": 100}]))
    got = ocr.ollama_offload("qwen3-vl:2b-instruct-q8_0")
    assert got is not None and got[0] == pytest.approx(1.0)


def test_a_zero_size_is_refused_rather_than_divided_by(monkeypatch):
    _patch_urlopen(monkeypatch, _ps([{"name": "m", "size": 0, "size_vram": 0}]))
    assert ocr.ollama_offload("m") is None


# ------------------------------------------------------------- the reader

def _reader(monkeypatch, frac):
    r = ocr.OllamaReader.__new__(ocr.OllamaReader)
    r.url, r.model, r.timeout = "http://x", "m", 1.0
    monkeypatch.setattr(
        ocr, "ollama_offload",
        lambda *a, **k: None if frac is None else (frac, 1.0e9, 2.0e9))
    return r


def test_a_fast_read_says_nothing_however_it_is_offloaded(monkeypatch):
    """The guard against crying wolf.

    58% on the GPU measured 0.57 s/label and finished the drawing in 15 s.
    Warning there would have been a false alarm, and the next real warning
    would have been ignored because of it.
    """
    for frac in (1.0, 0.58, 0.38, None):
        r = _reader(monkeypatch, frac)
        assert r.why_slow(0.6) is None, frac


def test_a_slow_read_is_explained_and_fixable(monkeypatch):
    r = _reader(monkeypatch, 0.38)
    note = r.why_slow(10.2)
    assert note is not None
    assert "10.2s each" in note
    assert "38% of the model is on the GPU" in note
    assert "keep_alive" in note, "the note must say how to fix it"


def test_a_slow_read_still_reports_when_ollama_will_not_say(monkeypatch):
    """If /api/ps cannot be reached the slowness is still real and still
    worth saying; only the cause is missing."""
    r = _reader(monkeypatch, None)
    note = r.why_slow(10.2)
    assert note is not None and "10.2s each" in note
    assert "on the GPU" not in note


def test_the_threshold_sits_between_the_two_measured_regimes():
    assert 0.8 < ocr.SLOW_LABEL_SECONDS < 10.0


# ------------------------------------------------------- the whole plumbing

def test_a_slow_reading_reaches_info_so_the_pipeline_can_warn(monkeypatch):
    """`read_structured` measures, the reader explains, `info` carries it.

    The pipeline turns `info["slow"]` into a graph warning, which is what the
    CLI prints and the GUI shows as a note. Without this path the slowness is
    real and invisible, which is how it went unnoticed in the first place.
    """
    import numpy as np

    from circuit2spice.graph import Detection

    class _Slow:
        model = "stub"

        def read(self, crop, letter=None, det=None, with_nets=False, **k):
            return "R1"

        def why_slow(self, spl):
            return f"slow: {spl:.1f}s" if spl >= ocr.SLOW_LABEL_SECONDS else None

    monkeypatch.setattr(ocr, "make_reader", lambda *a, **k: _Slow())
    # Pretend every read took four seconds, without actually waiting.
    clock = iter([0.0, 4.0] * 20)
    monkeypatch.setattr("time.time", lambda: next(clock))

    dets = [Detection("text", 10, 10, 60, 30),
            Detection("resistor", 10, 40, 60, 90)]
    info: dict = {}
    n = ocr.read_structured(np.zeros((200, 200, 3), np.uint8), dets, info=info)
    assert n == 1
    assert info["seconds_per_label"] == pytest.approx(4.0)
    assert info["slow"].startswith("slow: 4.0s")


def test_a_fast_reading_leaves_info_clean(monkeypatch):
    import numpy as np

    from circuit2spice.graph import Detection

    class _Fast:
        model = "stub"

        def read(self, crop, letter=None, det=None, with_nets=False, **k):
            return "R1"

        def why_slow(self, spl):
            return None

    monkeypatch.setattr(ocr, "make_reader", lambda *a, **k: _Fast())
    dets = [Detection("text", 10, 10, 60, 30),
            Detection("resistor", 10, 40, 60, 90)]
    info: dict = {}
    ocr.read_structured(np.zeros((200, 200, 3), np.uint8), dets, info=info)
    assert "slow" not in info
