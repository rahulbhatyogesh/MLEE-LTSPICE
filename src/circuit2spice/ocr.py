"""Closed-vocabulary reading of the labels beside each symbol.

Free-running a recogniser over these crops does not work, and the measurements
say why. ``easyocr`` returns ``Cour`` for ``Cout`` and nothing at all for half
the boxes: its recogniser is a CRNN trained on printed and scene text. TrOCR
does better -- ``tooof`` for ``1000F``, ``ton`` for ``10ohm``, ``ci`` for
``C1``, so it *is* reading the strokes -- but its decoder is a language model
trained on English prose, so it rounds every reading towards an English word
and lower-cases on the way. Repairing that afterwards with a confusion map and
a grammar filter reached 5 labels of 15, and forcing the nearby component's
letter made the value-only boxes worse: ``1000F`` became ``C000F``.

The mistake was the question. Free generation asks *"what does this say?"*,
whose answer space is every string in English. A drawing with a naming
convention lets us ask *"which of these does it say?"* instead:

- a designator is a class letter and an index, and the **detector already says
  the class**, because it found the symbol the label sits beside;
- a value is a number, an optional SI prefix and a unit, and the same letter
  fixes which units are legal.

So the grammar stops being a filter applied after the damage and becomes the
search space itself. Every legal label is scored under the image and the
highest log-likelihood wins, which means the output cannot be unparseable: the
worst case is the wrong legal label rather than ``mystiques``.

**Candidates must be written the way the model writes.** TrOCR ends every line
with a space and a full stop, and a candidate without one is scored in a
surface form its decoder never produces. The effect is not subtle -- on a
clean synthetic image of the word *hello*, the candidate ``hello`` scores
*below* ``zebra``, while ``hello .`` beats it. Appending the suffix is what
makes the ranking mean anything.

Scoring the full product of designators and values is exact but costs one
forward pass per combination. Scoring the two parts separately costs the sum
instead: each is penalised for the other's pixels, but by the same amount for
every candidate of its kind, so the ranking within each part survives.
"""

from __future__ import annotations

import itertools
import os
import re

import numpy as np

from .classes import PLACEABLE
from .values import DESIGNATOR_CLASSES, _gap, parse_label

#: Preferred-value series plus the round numbers people actually write.
MANTISSAS = [
    "1", "1.5", "2", "2.2", "3", "3.3", "4", "4.7", "5", "6", "6.8", "8", "9",
    "10", "15", "22", "33", "47", "56", "68", "100", "150", "220", "330",
    "470", "680", "1000", "2200", "4700",
]

#: Units each class may carry, spelled the ways people write them.
UNITS: dict[str, list[str]] = {
    "R": ["Ω", " ohms", " ohm"],
    "C": ["F", "pF", "nF", "uF", "µF"],
    "L": ["H", "mH", "uH"],
    "V": ["V"],
    "I": ["A", "mA"],
}

#: SI prefixes worth spelling out per class. Capacitors and inductors carry
#: theirs inside the unit (``pF``, ``mH``), so they take none here.
PREFIXES: dict[str, list[str]] = {
    "R": ["", "k", "K", "M"], "C": [""], "L": [""], "V": [""], "I": [""],
}

#: Index part of a designator. Numbers are always legal; the word suffixes
#: belong to a signal at the edge of a drawing, so ``Rout`` is only plausible
#: beside a terminal. Offering it everywhere cost 4 of 15 labels on the test
#: drawing -- it is short, so a mean log-probability likes it, and it beat the
#: true reading on four resistor boxes that had nothing to do with an output.
INDICES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
WORD_INDICES = ["in", "out"]

#: What TrOCR puts at the end of every line it reads.
SURFACE_SUFFIX = " ."

#: Component class -> its designator letter.
CLASS_LETTER: dict[str, str] = {}
for _letter, _classes in DESIGNATOR_CLASSES.items():
    for _c in _classes:
        CLASS_LETTER.setdefault(_c, _letter)

MODEL = os.environ.get("CIRCUIT2SPICE_TROCR", "microsoft/trocr-base-handwritten")


#: Names for a *wire*, legal beside any symbol. These must be in every pool:
#: ``Vout`` sits nearest a capacitor on one drawing, and a pool built only
#: from that capacitor's class cannot contain it, so the scorer is forced to
#: return a capacitance for a box that plainly says Vout.
NET_LABELS = ["Vin", "Vout", "GND", "VCC", "VDD", "VSS", "CLK"]


def designators(letter: str, with_nets: bool = True) -> list[str]:
    out = [f"{letter}{i}" for i in INDICES]
    if with_nets:
        out += [f"{letter}{i}" for i in WORD_INDICES] + NET_LABELS
    return out


def values(letter: str) -> list[str]:
    out = []
    for m, p, u in itertools.product(MANTISSAS, PREFIXES.get(letter, [""]),
                                     UNITS.get(letter, [""])):
        out.append(f"{m}{p}{u}")
    return out


#: Rough width of one hand-printed character, as a fraction of its height.
#: Used only to estimate how many characters a box can be holding.
CHAR_ASPECT = 0.46

#: How far a candidate's length may stray from that estimate.
LENGTH_SLACK = 1.5


def plausible_length(det, cands: list[str]) -> list[str]:
    """Keep the candidates whose length suits the box's shape.

    Scoring alone is biased towards short strings -- a mean log-probability
    rewards a candidate for having few tokens to be wrong about, so ``Rout``
    beats ``R3 10 ohms`` on a box that plainly contains both a name and a
    value. The box itself settles it: a designator-only label is about as wide
    as it is tall, a designator with a value is three times wider. That is
    geometry we already have and the recogniser cannot see.

    Never returns an empty list -- a bad estimate must not silence a box.
    """
    if det.h <= 0:
        return cands
    want = (det.w / det.h) / CHAR_ASPECT
    keep = [c for c in cands if abs(len(c) - want) <= LENGTH_SLACK]
    return keep or cands


def beside_a_terminal(text_det, detections: list) -> bool:
    """Is the nearest symbol to this label a terminal rather than a part?

    ``Vout`` names a wire and is written at a terminal; ``R3 10 ohms`` names a
    part. Offering the net labels everywhere lets ``Rout`` and ``VDD`` win
    boxes that plainly hold a resistor value -- measured as 3 of 15 labels
    lost that way -- so they are only candidates where a net label belongs.
    """
    near = None
    best = None
    for d in detections:
        if d.cls in ("text", "junction", "crossover"):
            continue
        g = _gap(text_det, d)
        if best is None or g < best:
            near, best = d, g
    return near is not None and near.cls in ("terminal", "gnd")


def nearest_letter(text_det, components: list) -> str | None:
    """The designator letter of the component this label sits nearest."""
    if not components:
        return None
    near = min(components, key=lambda c: _gap(text_det, c))
    return CLASS_LETTER.get(near.cls)


class Reader:
    """Scores candidate strings against a crop. Loads the model once."""

    def __init__(self, model_name: str | None = None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        name = model_name or MODEL
        self.proc = TrOCRProcessor.from_pretrained(name)
        self.model = VisionEncoderDecoderModel.from_pretrained(name)
        self.model.eval()
        self.tok = self.proc.tokenizer
        # transformers 5 keeps this on the generation config, not the model
        # config; reading the wrong one raises AttributeError on the first crop.
        self.start = self.model.generation_config.decoder_start_token_id

    def encode(self, crop: np.ndarray):
        from PIL import Image

        rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
        pv = self.proc(images=Image.fromarray(np.ascontiguousarray(rgb)),
                       return_tensors="pt").pixel_values
        with self.torch.no_grad():
            return self.model.encoder(pv).last_hidden_state

    def score_all(self, enc, cands: list[str], batch: int = 12):
        """``[(candidate, mean log-prob per token)]``, best first."""
        torch = self.torch
        out: list = []
        with torch.no_grad():
            for i in range(0, len(cands), batch):
                chunk = [c + SURFACE_SUFFIX for c in cands[i:i + batch]]
                lab = self.tok(chunk, return_tensors="pt", padding=True).input_ids
                din = torch.cat(
                    [torch.full((lab.shape[0], 1), self.start, dtype=torch.long),
                     lab[:, :-1]], dim=1)
                hs = enc.expand(lab.shape[0], -1, -1)
                logits = self.model(encoder_outputs=(hs,),
                                    decoder_input_ids=din).logits
                lp = torch.log_softmax(logits.float(), -1)
                tokl = lp.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
                mask = (lab != self.tok.pad_token_id).float()
                # Mean, not sum: a sum always prefers the shortest candidate.
                s = (tokl * mask).sum(1) / mask.sum(1).clamp(min=1)
                out.extend(zip(cands[i:i + batch], s.tolist()))
        out.sort(key=lambda t: -t[1])
        return out

    def score(self, enc, cands: list[str], batch: int = 12):
        """``(best candidate, mean log-prob per token)``."""
        top = self.score_all(enc, cands, batch)
        return top[0] if top else ("", -1e9)

    def read(self, crop: np.ndarray, letter: str, det=None,
             with_nets: bool = True, top_k: int = 5) -> str:
        """The most likely legal label for this crop, given the part beside it.

        A box carries a designator, a value, or both, and which one is not
        known in advance, so all three forms are scored and the likeliest
        wins. The combined form is built from the **top few** of each part,
        not the single best of each: when the two are scored separately each
        is penalised for the other's pixels, so the true designator is often
        second or third, and pairing only the winners propagates that error
        into the only candidate that could have been right.
        """
        enc = self.encode(crop)
        desig = designators(letter, with_nets=with_nets)
        vals = values(letter)
        if det is not None:
            desig = plausible_length(det, desig)
            vals = plausible_length(det, vals)

        d_top = self.score_all(enc, desig)[:top_k]
        v_top = self.score_all(enc, vals)[:top_k]
        best = []
        if d_top:
            best.append(d_top[0])
        if v_top:
            best.append(v_top[0])

        combos = [f"{a} {b}" for a, _ in d_top for b, _ in v_top]
        if det is not None:
            combos = [c for c in plausible_length(det, combos)]
        if combos:
            best.append(self.score(enc, combos))
        return max(best, key=lambda t: t[1])[0] if best else ""


#: A vision-language model can be *told* the convention. TrOCR cannot: its
#: decoder is a fixed language model trained on English prose, which is why it
#: answers "mystiques" to "R1 1k". Stating the grammar in a prompt is the
#: cheapest possible way to encode a naming convention, and it reads these
#: labels exactly where every TrOCR arrangement guessed.
#: The 2B model is the default because it is the one that was measured
#: end to end: 15/15 labels on the test drawing at 9.7 s each, against a
#: 3B run the OS killed for memory before it finished. ``CIRCUIT2SPICE_VLM``
#: selects a larger one where there is room for it.
VLM_MODEL = os.environ.get("CIRCUIT2SPICE_VLM", "Qwen/Qwen2-VL-2B-Instruct")

VLM_PROMPT = (
    "This is a label from a hand-drawn circuit diagram. It follows a strict "
    "convention: an optional reference like R1, C2, L3, Vin, Vout or Cout, "
    "then optionally a value like 10k ohms, 100 ohms, 10pF, 4F or 3H. "
    "Transcribe it exactly. Reply with the label only, no explanation. "
    # Without this the examples above become the fallback answer: asked to
    # read an illegible crop, the 8-bit reader returned "R1 10k ohms" -- a
    # phrase from this prompt, not from the drawing. An invented label is the
    # worst output there is, because nothing downstream can tell it from a
    # real one. Give the model somewhere else to go.
    "If you cannot read it, reply exactly: UNREADABLE."
)

#: What the prompt above offers instead of guessing.
NOT_READ = "UNREADABLE"

#: How to name a class to a reader, and the unit it is measured in.
_CLASS_WORDS = {"R": ("a resistor", "ohms"), "C": ("a capacitor", "farads"),
                "L": ("an inductor", "H (henries)"),
                "V": ("a voltage source", "volts"),
                "I": ("a current source", "amps"), "D": ("a diode", "no unit"),
                "Q": ("a transistor", "a part number"),
                "M": ("a transistor", "a part number")}


def prompt_for(letter: str | None, with_nets: bool = False) -> str:
    """The prompt, plus the one thing the detector knows that helps.

    The scorer uses the class to *build* its candidate pool; a VLM has to be
    told. Measured on a label the drawing writes as ``L1 3H``, whose L and 1
    are joined into a single stroke: unprompted the reader returns ``U3N``,
    and told the unit it returns ``L3H``, from which the henries survive.

    **Only the unit, never the reference letter.** Naming the letter cost
    four labels of fifteen: told "a reference starts with R", the reader
    prepends one to a value that has none -- ``10Ω`` came back as ``R10Ω``
    and ``1pF`` as ``C10pF``. It is an invitation, not a constraint.

    The hint is dropped entirely next to a terminal, where the label may name
    a *net*: there the nearest component's class is unrelated to the answer,
    and ``Vout`` duly came back as ``Cout`` because the nearest thing to it
    is a capacitor.
    """
    words = _CLASS_WORDS.get((letter or "").upper())
    if not words or with_nets:
        return VLM_PROMPT
    return f"{VLM_PROMPT} If it includes a value, the unit is {words[1]}."


class VlmReader:
    """Reads a crop by asking a vision-language model, told the convention.

    Slower than scoring -- a few seconds per crop on CPU against a fraction of
    a second -- but it is the only approach measured here that returns the
    right string rather than the least-wrong legal one.
    """

    def __init__(self, model_name: str | None = None, max_pixels: int = 200_000,
                 dtype: str | None = None):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        name = model_name or VLM_MODEL
        # Cap the vision tokens: these crops are a few hundred pixels across,
        # and letting the processor scale them up is most of the runtime.
        self.proc = AutoProcessor.from_pretrained(name, max_pixels=max_pixels)
        # Let the checkpoint pick its own class. Naming one explicitly loads a
        # Qwen2-VL checkpoint into the Qwen2.5-VL class, which "succeeds" with
        # a shape-mismatch report and then reads nothing at all.
        from transformers import AutoModelForImageTextToText as Cls
        # float32 is what the reference snippets use and it needs roughly
        # 14 GB for a 3B model -- enough that the OS killed the first run of
        # this on a 16 GB machine. bfloat16 halves that and costs nothing
        # measurable in transcription quality, so it is the default here.
        want = dtype or os.environ.get("CIRCUIT2SPICE_VLM_DTYPE", "bfloat16")
        try:
            self.model = Cls.from_pretrained(name, dtype=getattr(torch, want))
        except (TypeError, ValueError, RuntimeError):
            self.model = Cls.from_pretrained(name, dtype=torch.float32)
        self.model.eval()

    def read(self, crop: np.ndarray, letter: str | None = None, det=None,
             with_nets: bool = False, **_ignored) -> str:
        from PIL import Image

        rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
        pil = Image.fromarray(np.ascontiguousarray(rgb))
        msgs = [{"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text",
                              "text": prompt_for(letter, with_nets)}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[pil], return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=24, do_sample=False)
        got = self.proc.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0]
        # Same cleaning as the Ollama path: one prompt, one set of surface
        # conventions, so a reading cannot mean two different things.
        return clean_reading(got)


#: Ollama holds the model in its *own* process, which is the whole reason to
#: prefer it: the weights land in VRAM rather than on this process's heap, and
#: a 4-bit build is small enough to fit a card that could not hold the same
#: model in bfloat16 -- 3.3 GB for a 4B against the 4.4 GB our 2B costs here.
#: It also means a reading failure cannot take the pipeline's memory with it.
OLLAMA_URL = os.environ.get("CIRCUIT2SPICE_OLLAMA", "http://127.0.0.1:11434")

#: Measured, not chosen by size. On the 15-label drawing, 8-bit 2B reads 14,
#: 4-bit 4B reads 12 and 4-bit 2B reads 11, so **quantisation decides this,
#: not parameter count** -- the labels it recovers are exactly the omega-glyph
#: ones, and 4 bits is what blurs omega against 0 and n. The plain ``:2b`` and
#: ``:4b`` tags are *thinking* builds whose answer never reaches ``response``
#: at all; always name an ``-instruct`` tag.
#: ``CIRCUIT2SPICE_OLLAMA_MODEL`` overrides it.
OLLAMA_MODEL = os.environ.get("CIRCUIT2SPICE_OLLAMA_MODEL",
                              "qwen3-vl:2b-instruct-q8_0")


def ollama_models(url: str | None = None, timeout: float = 1.0) -> list[str]:
    """Names the Ollama service is holding, or ``[]`` if it is not running.

    The probe has to be cheap and silent, because "no service" is the normal
    case on a machine that never installed one and must cost nothing.
    """
    import json
    import urllib.request

    base = (url or OLLAMA_URL).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=timeout) as resp:
            got = json.load(resp)
    except Exception:
        return []
    return [m.get("name", "") for m in got.get("models", []) if m.get("name")]


#: A label that takes longer than this is coming off the CPU. Measured on the
#: 4 GB RTX 3050: fully or mostly resident reads land at 0.4-0.8 s warm, and a
#: heavily split model at ~10 s. Three seconds is a gap, not a line drawn
#: through a cluster.
#:
#: The trigger is *measured time*, deliberately, not the offload fraction. The
#: fraction is tempting and wrong: 58% on the GPU still reads at 0.57 s/label
#: and the whole drawing still finishes in 15 s, so a threshold set on it
#: fires when nothing is wrong. A warning that cries wolf is worse than no
#: warning, because the next real one gets ignored too.
SLOW_LABEL_SECONDS = 3.0


def ollama_offload(model: str, url: str | None = None, timeout: float = 2.0):
    """How much of a loaded model Ollama actually put on the GPU.

    Returns ``(fraction, vram_bytes, total_bytes)``, or ``None`` when the
    model is not loaded or the service cannot be asked.

    This exists because a partial offload is **silent and expensive**. Ollama
    decides the split once, when the model loads, from the VRAM free at that
    moment -- and never revisits it. Train something on the same card, or run
    any other GPU job, and the reader that loads during that window stays half
    on the CPU for the rest of its life. The pipeline still produces exactly
    the right answer, just ten times slower, which is the hardest kind of
    regression to notice and the easiest to blame on the wrong change.
    """
    import json
    import urllib.request

    base = (url or OLLAMA_URL).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/ps", timeout=timeout) as resp:
            got = json.load(resp)
    except Exception:
        return None
    for m in got.get("models", []):
        name = m.get("name", "")
        if not (name == model or name.startswith(model) or model.startswith(name)):
            continue
        total = int(m.get("size") or 0)
        vram = int(m.get("size_vram") or 0)
        if total <= 0:
            return None
        return vram / total, vram, total
    return None


def ollama_payload(model: str, prompt: str, crop: np.ndarray) -> dict:
    """The request body, kept pure so it can be checked without a service.

    The crop goes as-is. Upscaling it to 160 px tall first was the obvious
    suspect for the accuracy gap against the local reader, which is handed
    these crops 3x upscaled -- but it measured *worse*, 10/15 against 11/15
    with ``10Ω`` degrading to ``10r2``, so the gap is the model, not the
    resolution. Don't re-derive it.
    """
    import base64

    import cv2

    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("could not encode the crop")
    return {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(buf.tobytes()).decode("ascii")],
        "stream": False,
        # Thinking off. The default ``qwen3-vl:2b`` tag is a *thinking* build,
        # and Ollama puts that reasoning in a separate ``thinking`` field --
        # so the answer never reaches ``response`` at all: the budget goes on
        # "So, let's look at the image. The text is \"Cout\". Wait," and the
        # reply comes back empty with ``done_reason: length``. Measured: every
        # one of 16 boxes read as "" until this flag went in.
        "think": False,
        # Temperature 0. A transcription has one right answer, and sampling is
        # exactly how a reader invents a *plausible* wrong label -- which is
        # the one failure mode this module exists to avoid.
        "options": {"temperature": 0.0, "num_predict": 48},
    }


def clean_reading(text: str) -> str:
    """One label out of whatever the model decided to say.

    A ``-thinking`` build reasons out loud before answering and an instruct
    build sometimes wraps the answer in a sentence, so the *last* non-empty
    line wins and the usual decorations come off. Nothing here invents a
    label: unreadable input still returns the empty string.
    """
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = text.replace("```", "\n")
    lines = [ln.strip().strip("`\"'*") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    out = lines[-1]
    # "The label reads: R1 1k" -- no circuit label contains a colon, so
    # anything before the last one is the model talking about its answer.
    if ":" in out:
        tail = out.rsplit(":", 1)[1].strip()
        if tail:
            out = tail
    out = out.strip().strip("`\"'*").rstrip(" .")
    return "" if out.upper() == NOT_READ else out


class OllamaReader:
    """Reads a crop through a local Ollama service.

    Same prompt as :class:`VlmReader` -- the convention stated in words --
    against a newer and more strongly quantised model, run on the GPU by a
    process that is not ours.
    """

    def __init__(self, model_name: str | None = None, url: str | None = None,
                 timeout: float = 120.0):
        self.url = (url or OLLAMA_URL).rstrip("/")
        self.model = model_name or OLLAMA_MODEL
        self.timeout = timeout
        held = ollama_models(self.url)
        if not held:
            raise RuntimeError(f"no Ollama service answering at {self.url}")
        # ``qwen3-vl:4b`` and ``qwen3-vl:4b-instruct-q4_K_M`` are one build
        # under two names, so a prefix match is what "is it pulled" means.
        if not any(h == self.model or h.startswith(self.model) for h in held):
            raise RuntimeError(f"{self.model} is not pulled (have: {', '.join(held)})")

    def read(self, crop: np.ndarray, letter: str | None = None, det=None,
             with_nets: bool = False, **_ignored) -> str:
        import urllib.error

        payload = ollama_payload(self.model, prompt_for(letter, with_nets), crop)
        try:
            got = self._post(payload)
        except urllib.error.HTTPError as exc:
            # A build with no thinking to disable rejects the flag outright;
            # drop it and ask again rather than losing the reading.
            if exc.code != 400 or "think" not in payload:
                raise
            payload.pop("think")
            got = self._post(payload)
        text = clean_reading(got.get("response") or "")
        if not text and got.get("thinking"):
            # Loud on purpose. A reader that returns nothing is indis-
            # tinguishable from illegible handwriting, and that cost an
            # afternoon once already.
            raise RuntimeError(
                f"{self.model} answered only in its thinking field; pull an "
                f"-instruct tag (e.g. {self.model}-instruct) instead")
        return text

    def why_slow(self, seconds_per_label: float) -> str | None:
        """Explain a slow read, or stay quiet.

        Called with the *measured* cost per label once the reading is done.
        Only then is it worth asking ``/api/ps`` how much of the model Ollama
        put on the GPU, because only then do we know there is something to
        explain. The offload is reported as the likely cause rather than the
        trigger -- see :data:`SLOW_LABEL_SECONDS`.
        """
        if seconds_per_label < SLOW_LABEL_SECONDS:
            return None
        msg = (f"label reading took {seconds_per_label:.1f}s each "
               f"({self.model}); it is normally under 1s")
        got = ollama_offload(self.model, self.url)
        if got is not None:
            frac, vram, total = got
            msg += (f". Only {frac * 100:.0f}% of the model is on the GPU "
                    f"({vram / 1e9:.1f} of {total / 1e9:.1f} GB), so the rest "
                    f"is running on the CPU. Ollama fixes that split when the "
                    f"model loads and never revisits it, so it happens when "
                    f"something else held the card at that moment -- training "
                    f"the diode or U-Net models, for instance")
        msg += (f". Free the GPU and reload it: curl -X POST "
                f"{self.url}/api/generate "
                f'-d \'{{"model":"{self.model}","keep_alive":0}}\'')
        return msg

    def _post(self, payload: dict) -> dict:
        import json
        import urllib.request

        req = urllib.request.Request(
            self.url + "/api/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)


def make_reader(backend: str = "auto", model_name: str | None = None):
    """The best reader installed, or ``None``.

    Order is a measured preference, not a taste: Ollama keeps the weights out
    of this process and on the GPU, the local ``vlm`` reads as well but holds
    several GB here, and the scorer always works but returns the least-wrong
    *legal* string rather than the right one.
    """
    if backend in ("auto", "ollama"):
        try:
            # Only an explicit ``ollama`` backend may name the model: under
            # ``auto`` the name belongs to the transformers checkpoint and is
            # not a tag this service would recognise.
            return OllamaReader(model_name if backend == "ollama" else None)
        except Exception:
            if backend == "ollama":
                return None
    if backend in ("auto", "vlm"):
        try:
            return VlmReader(model_name)
        except Exception:
            if backend == "vlm":
                return None
    if backend in ("auto", "scored"):
        try:
            return Reader(model_name)
        except Exception:
            return None
    return None


#: Letters a recogniser writes where a digit belongs.
_LOOKALIKES = str.maketrans({"s": "5", "S": "5", "o": "0", "O": "0", "i": "1",
                             "l": "1", "I": "1", "z": "2", "Z": "2", "b": "6",
                             "g": "9", "q": "9"})

#: The unit each class is measured in, for repairing a mangled one.
_UNIT_OF = {"R": "Ω", "C": "F", "L": "H", "V": "V", "I": "A"}


#: ``_UNIT_OF`` again, in the spelling :func:`values.parse_value` canonicalises
#: to, so a bare value can be ranked as the quantity it is.
_ASSUMED_UNIT = {"Ω": "ohm", "F": "F", "H": "H", "V": "V", "A": "A"}


def _plausibility(text: str, unit: str | None = None) -> float:
    """How much a parsed reading looks like a label someone would write.

    ``unit`` is the symbol the neighbouring component implies, and leaving it
    out is not neutral: a label written as a bare ``1k`` -- which is how most
    people write one -- parses as nothing without it, scores ``-1e9``, and is
    beaten by every candidate that merely parses, including the ones that
    threw a character away to get there.
    """
    desig, value = parse_label(text, assume_unit=_ASSUMED_UNIT.get(unit or ""))
    if not desig and not value:
        return -1e9
    score = 0.0
    if desig:
        index = desig[1:]
        # R3 is a part; R310 is a misread omega glued onto R3 and 10.
        score += 2.0 if (not index.isdigit() or len(index) == 1) else -2.0
    if value is not None:
        score += 1.0
        # A sub-ohm resistor is not something anyone draws, and that is the
        # tie-break the omega glyph needs: ``R4 1n`` parses perfectly well as
        # one nanoohm, scores exactly as well as ``R4 1Ω`` on every other
        # term, and wins on being first. Measured: the 4-bit readers return
        # ``R4 1n`` and ``6n`` for ``R4 1Ω`` and ``6Ω``.
        # Only resistance is checked. A drawing that says ``C2 10F`` means ten
        # farads however absurd that capacitor would be, so there is no
        # equivalent prior to apply on the other classes.
        if value.unit == "ohm" and value.magnitude is not None                 and 0 < value.magnitude < 1.0:
            score -= 3.0
    return score


def repair_reading(text: str, letter: str | None) -> str:
    """Fix the handful of glyphs a recogniser reliably gets wrong.

    Measured on one drawing, every residual error was the same thing: the ohm
    sign came back as ``r``, ``n``, ``2`` or ``0`` -- ``10ohm`` as ``10r``,
    ``R4 1ohm`` as ``R410`` -- plus ``R5`` as ``Rs``. Those are shape
    confusions, not reading failures.

    A repair is only accepted if it *parses*, and the candidates are ranked
    rather than taken first-come. That ordering matters: ``R3102`` already
    parses, as designator ``R310`` with the value 2, so a first-match rule
    keeps the wrong reading. Preferring a single-digit index picks the repair
    ``R310ohm``, which parses as ``R3`` with 10 ohms.
    """
    s = " ".join((text or "").split())
    if not s:
        return s
    cands = [s]

    def add(c):
        if c and c not in cands:
            cands.append(c)

    add(s[0] + s[1:].translate(_LOOKALIKES))
    add(s.translate(_LOOKALIKES))
    unit = _UNIT_OF.get(letter or "")
    if unit:
        for base in list(cands):
            if len(base) > 1 and base[-1] not in "0123456789":
                add(base[:-1] + unit)          # a mangled unit glyph
            if len(base) > 1:
                add(base[:-1] + unit)          # a unit read as a digit
            add(base + unit)                   # a unit dropped entirely
    # A mangled unit does not always arrive as a look-alike character: the
    # 8-bit reader returns ``R1 1k-52`` for ``R1 1kΩ``, with the glyph
    # expanded into junk rather than substituted. Trimming the tail is safe
    # only because it goes through the same ranking -- ``max`` returns the
    # *first* maximal candidate and the original is first, so a trim can only
    # win when the reading as given does not parse at all.
    # ... but never off a reading that already ends in the right unit. There
    # the last character is the most informative one in the string: trimming
    # ``L3H`` to ``L3`` throws the henries away and invents a part number,
    # and a designator outranks a value, so it would win.
    if not (unit and s.endswith(unit)):
        for k in range(1, 5):
            if len(s) - k < 2:
                break
            add(s[:-k])
            if unit:
                add(s[:-k] + unit)
    # ``L3H``: the class letter survived but the index did not, so nothing
    # parses at all -- not even the value, which is the half that matters.
    # Dropping the letter leaves ``3H``. Ranking keeps this honest: ``R5``
    # would become the unparseable ``5`` and loses to the reading as given.
    # The letter need not be the *right* one: the drawing says ``L1 3H`` and
    # the reader returns ``U3H``, having joined the L and the 1 into one
    # stroke. Any letter followed directly by a digit is a candidate prefix.
    if len(s) > 1 and s[0].isalpha() and s[1].isdigit():
        add(s[1:].strip())
    # A stroke of the symbol caught in the crop, read as punctuation. The
    # capacitor's own plate comes back as a leading hyphen -- ``10n`` reads as
    # ``-10n``, ``10u`` as ``-10u``, ``1k`` as ``.1k`` -- and none of those
    # parse at all, so a value the reader got exactly right is discarded and
    # the part silently keeps its class default. Three of the labels on one
    # drawing went this way.
    #
    # Safe because ``_VALUE_RE`` anchors on a digit, so a reading that starts
    # with punctuation parses as nothing whatever we do, and the ranking only
    # accepts a candidate that parses. A genuine negative value is not lost
    # here because it was never supported in the first place.
    stripped = s.lstrip("-–—.,:;_~^'· ")
    if stripped and stripped != s:
        add(stripped)
        if stripped[0].isalpha() and len(stripped) > 1 and stripped[1].isdigit():
            add(stripped[1:].strip())
    best = max(cands, key=lambda c: _plausibility(c, unit))
    return best if _plausibility(best, unit) > -1e8 else s


def complete_unit(text: str, letter: str | None) -> str:
    """Write the implied unit onto a reading that is only a bare value.

    ``1k`` beside a resistor means a kilohm, and the drawing simply does not
    repeat the symbol -- people write ``R1`` on one line and ``1k`` on the
    next. Everything downstream keys off the unit, though: it is the unit that
    decides which class a value may belong to, so a value with none attached
    is dropped on the floor rather than assigned to the wrong part.

    So the letter that was already trusted to build the prompt and rank the
    repair is trusted once more, here, to finish the string. Only a reading
    that parses as a value *solely* because of the assumption is touched; a
    label that already says what it is, or that is a designator, is left
    exactly as it was read.
    """
    unit = _UNIT_OF.get((letter or "").upper())
    if not unit or not text:
        return text
    if parse_label(text) != (None, None):
        return text                       # already complete; do not meddle
    desig, value = parse_label(text, assume_unit=_ASSUMED_UNIT.get(unit))
    if desig or value is None:
        return text
    return f"{text}{unit}"


def reader_name(reader) -> str:
    """How to describe a reader to someone watching the GUI."""
    model = getattr(reader, "model", None) or getattr(reader, "name", "")
    kind = {"OllamaReader": "Ollama", "VlmReader": "local",
            "Reader": "closed-vocabulary scoring"}.get(
                type(reader).__name__, type(reader).__name__)
    return f"{model} ({kind})" if model else kind


def read_structured(image, detections: list, model_name: str | None = None,
                    backend: str = "auto", info: dict | None = None) -> int:
    """Fill in ``Detection.text`` for every text box, by scoring.

    Returns how many were read. Only boxes with a component near enough to
    name a class are attempted: a label we cannot constrain is one we would be
    guessing at, and everything downstream handles a missing value better than
    a wrong one.

    ``info``, if given, collects ``reader`` -- which of three backends
    actually ran. Worth surfacing: they differ by 25x in speed and one label
    in accuracy, and "auto" means the answer depends on whether a service
    happens to be up. It also collects ``slow``, set when the Ollama reader
    turns out to be running half on the CPU, which costs the same 10x without
    changing a single character of the output.
    """
    boxes = [d for d in detections if d.cls == "text" and not getattr(d, "text", "")]
    comps = [d for d in detections if d.cls in PLACEABLE]
    if not boxes or not comps:
        return 0

    reader = make_reader(backend, model_name)
    if reader is None:
        return 0
    if info is not None:
        info["reader"] = reader_name(reader)
    errors: list[str] = []
    h, w = image.shape[:2]
    n = 0
    import time as _time
    spent = 0.0
    for d in boxes:
        letter = nearest_letter(d, comps)
        if letter is None:
            continue
        # Pad by the box's *height*: that is the character size, and detector
        # boxes clip horizontally -- "Vin" comes back boxed from the i onward,
        # so no reader could return more than "in".
        px = max(3, int(0.35 * d.h))
        py = max(2, int(0.15 * d.h))
        x1, y1 = max(0, int(d.x1) - px), max(0, int(d.y1) - py)
        x2, y2 = min(w, int(d.x2) + px), min(h, int(d.y2) + py)
        if x2 <= x1 or y2 <= y1:
            continue
        try:
            t0 = _time.time()
            raw = reader.read(image[y1:y2, x1:x2], letter, det=d,
                              with_nets=beside_a_terminal(d, detections))
            spent += _time.time() - t0
            d.text_raw = raw
            d.text = complete_unit(repair_reading(raw, letter), letter)
            n += 1
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            # Swallowing these silently once cost an entire debugging round:
            # a reader that failed to load looked exactly like a drawing with
            # no legible labels. Keep going, but say what went wrong.
            if not errors:
                errors.append(f"{type(e).__name__}: {e}")
            continue
    if errors and not n:
        raise RuntimeError(f"no labels could be read -- first error was {errors[0]}")
    # Asked after the reads, and only of a reader that offers an opinion:
    # the question is "that was slow, why?", which cannot be put before there
    # is a measurement to put it about.
    if n and info is not None and hasattr(reader, "why_slow"):
        info["seconds_per_label"] = spent / n
        note = reader.why_slow(spent / n)
        if note:
            info["slow"] = note
    return n


def rerepair(detections: list) -> int:
    """Repair every stored reading again, against the classes as they are now.

    The repair is unit-dependent, and the unit comes from whichever component
    the label sits beside. Correct that component's class -- ``--as
    R10=inductor`` -- and the repair that was applied is simply the wrong one:
    the drawing's ``100u`` was turned into ``100ohm`` by the sub-ohm prior,
    because a 100 microohm resistor is not a thing, while 100 microhenries is
    exactly what a boost converter wants.

    Nothing is re-read here; the model is not asked again. Only the repair is
    redone, from the raw string the reader returned, which is why
    :attr:`Detection.text_raw` is kept.
    """
    comps = [d for d in detections if d.cls in PLACEABLE]
    if not comps:
        return 0
    n = 0
    for d in detections:
        raw = getattr(d, "text_raw", "")
        if d.cls != "text" or not raw:
            continue
        letter = nearest_letter(d, comps)
        fixed = complete_unit(repair_reading(raw, letter), letter)
        if fixed != d.text:
            d.text = fixed
            n += 1
    return n
