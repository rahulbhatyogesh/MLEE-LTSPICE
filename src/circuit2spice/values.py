"""Read component values from the handwritten text beside each symbol.

Until now every component got a default -- R=1k, C=1u, L=1m -- and the only way
to correct one was ``--value R1=4.7k`` on the command line. The drawing says
what the values are; nothing read them. Every reference paper lists this as
future work.

The job splits in two, and only the second half needs a recogniser:

1. **Turn a string into a SPICE value.** ``10KΩ`` -> ``10k``, ``1µF`` -> ``1u``,
   ``2,2KΩ`` -> ``2.2k`` (a comma decimal, which one drafter used), ``2N2222``
   -> a transistor model name. Labels like ``VDD``, ``OUT`` or ``THRS`` are not
   values at all and must be rejected rather than guessed at.

2. **Decide which component a string belongs to.** This is where the unit earns
   its keep: ``10KΩ`` can only be a resistor, ``1µF`` can only be a capacitor.
   Nearest-neighbour alone is unreliable on a dense drawing -- C12 has 33 px
   between symbols -- but *nearest compatible* component is a much stronger
   constraint, because the nearest symbol in a crowded corner is usually of the
   wrong type.

CGHD's annotations carry the transcribed string in a ``<text>`` element, so with
ground-truth boxes this runs end to end today and measures the ceiling: how
much value accuracy is available *if* the handwriting is read perfectly. Wiring
in a real recogniser then only has to close the gap to that ceiling, and
:func:`values_from_detections` takes its text from ``Detection.text`` whatever
filled it in.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import cv2

from .classes import PLACEABLE, spec

#: Unit symbol -> the merged classes a value in that unit can belong to.
#: This is what makes assignment robust: a resistance cannot label a capacitor.
UNIT_CLASSES: dict[str, tuple[str, ...]] = {
    "ohm": ("resistor",),
    "F": ("capacitor",),
    "H": ("inductor",),
    "V": ("source_voltage",),
    "A": ("source_current",),
}

#: Handwritten SI prefix -> the suffix SPICE should see.
#:
#: SPICE is case-insensitive and reads ``M`` as *milli*, not mega, so a
#: handwritten ``10MΩ`` written through unchanged would come out a billion times
#: too small. Everything is therefore normalised to an unambiguous spelling:
#: ``meg`` for mega, ``m`` for milli, ``u`` for micro.
PREFIX_SPICE: dict[str, str] = {
    "k": "k", "K": "k",
    "M": "meg",          # uppercase M, handwritten, means mega
    "m": "m",            # lowercase m means milli
    "µ": "u", "μ": "u", "u": "u", "U": "u",
    "n": "n", "N": "n",
    "p": "p", "P": "p",
    "G": "g", "g": "g",
}

PREFIX_SCALE: dict[str, float] = {
    "k": 1e3, "meg": 1e6, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "g": 1e9,
}

#: Semiconductor part numbers: ``2N2222``, ``TIP41``, ``NE555``, ``1N4007``.
#: Two or more letters followed by digits, optionally with a leading digit.
_MODEL_RE = re.compile(r"^[0-9]?[A-Z]{1,3}[0-9]{2,5}[A-Z]?$")

_VALUE_RE = re.compile(
    r"^\s*"
    r"(?P<num>[0-9]+(?:[.,][0-9]+)?)"      # 10, 2.2, 2,2
    r"\s*"
    r"(?P<prefix>meg|MEG|[kKMmµμuUnNpPgG])?"
    r"\s*"
    r"(?P<unit>ohm|OHM|Ohm|[ΩωFfHhVvAa])?"
    r"\s*(?P<tail>[A-Za-z ]*)$"            # trailing "AC", "DC", stray spaces
)

_UNIT_CANON = {
    "Ω": "ohm", "ω": "ohm", "ohm": "ohm", "OHM": "ohm", "Ohm": "ohm",
    "F": "F", "f": "F",
    "H": "H", "h": "H",
    "V": "V", "v": "V",
    "A": "A", "a": "A",
}


@dataclass(frozen=True)
class ParsedValue:
    """One handwritten string, understood."""

    raw: str
    #: SPICE-ready text, e.g. ``10k``, ``2.2k``, ``1u``, ``2N2222``.
    spice: str
    #: ``ohm`` / ``F`` / ``H`` / ``V`` / ``A``, or ``None`` for a model name.
    unit: str | None
    #: Merged classes this value could belong to.
    classes: tuple[str, ...]
    #: Numeric magnitude in base units, or ``None`` for a model name.
    magnitude: float | None = None

    @property
    def is_model(self) -> bool:
        return self.unit is None


def parse_value(text: str | None, assume_unit: str | None = None) -> ParsedValue | None:
    """Understand one handwritten string, or return ``None`` if it is not a value.

    Rejects node labels (``VDD``, ``GND``, ``OUT``, ``THRS``, ``A``) rather than
    forcing them into a number: a wrong value is worse than a default, because
    the default is at least obviously a default.
    """
    if not text:
        return None
    s = text.strip()
    if not s:
        return None

    # A part number carries no unit and names a model, not a quantity.
    up = s.upper().replace(" ", "")
    if _MODEL_RE.match(up) and not re.match(r"^[0-9.,]+$", up):
        # Guard against "10V" style matches sneaking in: those have a unit.
        if not re.match(r"^[0-9]+(?:[.,][0-9]+)?[A-Za-z]$", up):
            return ParsedValue(raw=s, spice=up, unit=None,
                               classes=("transistor_bjt", "transistor_fet", "diode"))

    m = _VALUE_RE.match(s)
    if not m:
        return None
    unit_raw = m.group("unit")
    if not unit_raw:
        # No unit means no way to know which component it belongs to, and
        # guessing is how a 10 k resistor becomes a 10 F capacitor -- unless
        # the caller already knows, which it does when a designator was read
        # off the same label: the R in "R1 10k" says ohms as surely as a
        # written symbol would, and people routinely leave the symbol off.
        if assume_unit is None:
            return None
        # ...but only if the whole string was understood. ``tail`` exists to
        # absorb a trailing "AC"/"DC" that follows a unit already written, and
        # leaning on it here would swallow a *misread* unit instead: ``10r``
        # is the omega glyph come out wrong, and accepting it as "10, assumed
        # ohms, plus an r we are ignoring" makes a wrong reading parse, which
        # stops :func:`ocr.repair_reading` from ever preferring ``10Ω``.
        if (m.group("tail") or "").strip():
            return None
        unit = assume_unit
    else:
        unit = _UNIT_CANON.get(unit_raw)
    if unit is None:
        return None

    num = m.group("num").replace(",", ".")
    prefix_raw = m.group("prefix") or ""
    prefix = PREFIX_SPICE.get(prefix_raw, "meg" if prefix_raw.lower() == "meg" else "")

    try:
        magnitude = float(num) * PREFIX_SCALE.get(prefix, 1.0)
    except ValueError:
        return None

    # Trim a trailing ".0" so 10.0k prints as 10k.
    if num.endswith(".0"):
        num = num[:-2]
    return ParsedValue(
        raw=s,
        spice=f"{num}{prefix}",
        unit=unit,
        classes=UNIT_CLASSES.get(unit, ()),
        magnitude=magnitude,
    )


def _gap(text, comp) -> float:
    """Distance from the component's centre to the text box.

    Deliberately not edge-to-edge between the two boxes. A value label is wide,
    and on a dense drawing its far end often grazes the corner of a *different*
    symbol -- on C2 the ``10KΩ`` belonging to one resistor overlaps the box of
    the resistor below it, scoring a perfect 0 and stealing it. Measuring to the
    component's centre removes that: the label's length no longer buys it
    proximity to a symbol it merely reaches past.
    """
    dx = max(text.x1 - comp.cx, 0.0, comp.cx - text.x2)
    dy = max(text.y1 - comp.cy, 0.0, comp.cy - text.y2)
    return (dx * dx + dy * dy) ** 0.5


#: Which component classes a reference designator letter can name.
DESIGNATOR_UNITS: dict[str, str] = {
    "R": "ohm", "C": "F", "L": "H", "V": "V", "I": "A",
}

DESIGNATOR_CLASSES: dict[str, tuple[str, ...]] = {
    "R": ("resistor",),
    "C": ("capacitor",),
    "L": ("inductor",),
    "V": ("source_voltage",),
    "I": ("source_current",),
    "D": ("diode",),
    "Q": ("transistor_bjt",),
    "M": ("transistor_fet",),
}

#: Subscripts are how people actually write R1, and no recogniser emits them
#: as digits.
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉",
                            "0123456789")

#: A designator is a class letter plus an index: a number, or one of the
#: word suffixes people give to the signals at the edge of a drawing.
#: Deliberately excludes cc/dd/ss: VCC and VDD name a *rail*, not a part, and
#: letting them through renames whatever source happens to sit nearest.
_SUFFIXES = ("in", "out", "ref", "bias")
_DESIG_RE = re.compile(
    r"^(?P<letter>[RCLVIDQM])\s*_?\s*(?P<index>[0-9]{1,3}|"
    + "|".join(_SUFFIXES) + r")(?P<rest>.*)$", re.IGNORECASE)


def _is_bare_digit(value) -> bool:
    """A value of one digit and no multiplier: ``5``, not ``10`` or ``10k``."""
    return len(value.spice) < 2 and value.spice[-1:].isdigit()


def parse_label(text: str | None,
                assume_unit: str | None = None) -> tuple[str | None, "ParsedValue | None"]:
    """Split one handwritten string into ``(designator, value)``.

    ``assume_unit`` supplies the unit for a label that is *only* a value, the
    way the designator already supplies it for ``R1 10k``. It is the caller's
    job to know it -- the detector does, from the class of the component the
    box sits beside -- and passing it matters more than it looks: without it
    a bare ``1k`` parses as nothing at all, and a reading that parses as
    nothing loses to any rewrite of itself that parses as something. That is
    how ``1k`` beside a resistor was being "repaired" into ``1Ω``.

    A label on a circuit diagram is one of three things, and which one is not
    known in advance: a designator (``C1``, ``Cout``), a value (``1pF``), or
    both in one box (``R1 1kΩ``) -- the last being what a detector
    usually hands back, because people write them side by side. Parsing only
    values, as this module used to, throws away the designator *and* fails on
    the combined form, since ``1kΩ`` with ``R1`` glued to its front is
    not a number.

    Returns ``(None, None)`` for anything that is neither, which is the right
    answer for a node label such as ``GND``.
    """
    if not text:
        return None, None
    s = " ".join(text.translate(_SUBSCRIPTS).split())
    if not s:
        return None, None

    # Try designator-plus-tail before whole-string, or "C2 10F" is read as
    # the part number C210F: _MODEL_RE happily matches a letter-digit mix once
    # the space is gone, and a part number outranks a capacitance.
    m = _DESIG_RE.match(s)
    if m:
        letter, index = m.group("letter").upper(), m.group("index")
        rest = m.group("rest").strip(" :=–-")
        assumed = DESIGNATOR_UNITS.get(letter)
        if not rest:
            # Decided here, before the whole-string attempt, because
            # ``_MODEL_RE`` matches ``R10`` as a *part number*: a letter and
            # two digits is exactly the shape of a 2N2222. Every two-digit
            # designator in a drawing was therefore read as a model name and
            # handed to the nearest transistor as its value -- a MOSFET came
            # back as ``M1 ... R14``.
            return f"{letter}{index if index.isdigit() else index.lower()}", None
        if rest:
            value = parse_value(rest, assume_unit=assumed)
            if value is not None:
                return f"{letter}{index}", value
            # "R110k": R1 with 10k, or R11 with 0k? Prefer the split whose
            # remainder is a value, shortest index first.
            #
            # **Only when the two ran together.** A space after the index says
            # the index is finished, and splitting across one turns a
            # designator into a value: the reader returns ``R17 Ω`` for a
            # label reading ``R17 100k`` -- the index read, the value missed,
            # a stray ohm sign left behind -- and this loop cut ``17`` into
            # ``R1`` plus the value ``7Ω``. Four such labels on one drawing
            # produced R1 = 7, R1 = 2, R1 = 1, R1 = 0, which looks like a loop
            # counter leaking into the netlist and is nothing of the kind.
            spaced = m.group("rest")[:1].isspace()
            if not spaced and index.isdigit() and len(index) > 1:
                for cut in range(1, len(index)):
                    cand = parse_value(index[cut:] + rest, assume_unit=assumed)
                    # ...and the value it yields must carry a multiplier. The
                    # reader also returns ``R15Ω`` for ``R15 10k`` -- the
                    # index read, the value missed, a stray ohm sign left
                    # behind and no space to separate it -- and that splits
                    # just as happily into R1 plus 5 ohms. Every value on
                    # these drawings has a prefix (10k, 1k, 100k, 1u), so a
                    # bare single digit is the tell-tale of a cut designator.
                    if cand is not None and not _is_bare_digit(cand):
                        return f"{letter}{index[:cut]}", cand

    whole = parse_value(s, assume_unit=assume_unit)
    if whole is not None:
        return None, whole

    if m:
        letter, index = m.group("letter").upper(), m.group("index")
        rest = m.group("rest").strip(" :=–-")
        # A bare unit with no number is explicable: it is the symbol caught
        # by the reader where the value was not. ``R17 Ω`` is R17 with its
        # value unread, and keeping the designator is strictly better than
        # discarding the label -- the part then shows up in the "kept the
        # default value" warning rather than vanishing.
        if rest and _UNIT_CANON.get(rest) is None:
            return None, None        # a tail we cannot explain: trust nothing
        return f"{letter}{index if index.isdigit() else index.lower()}", None
    return None, None


#: A net label is a name for a *wire*, not a part: the thing a drafter writes
#: at the edge of a drawing so the signal can be referred to. Deliberately
#: permissive about case, because people write Vout, VOUT and vOut.
_NET_LABEL_RE = re.compile(
    r"^(v(?:in|out|cc|dd|ss|ref|bias)|gnd|vss|out|in|a|b|clk|clock|"
    r"sig|signal|node\s*[0-9]{1,2}|n[0-9]{1,3})$", re.IGNORECASE)


#: The one spelling each known net label normalises to. Without this, VOUT and
#: Vout are two different wires and the circuit quietly comes apart.
_NET_CANON = {
    "vin": "Vin", "in": "Vin", "vout": "Vout", "out": "Vout",
    "vcc": "VCC", "vdd": "VDD", "vss": "VSS", "gnd": "GND",
    "vref": "Vref", "vbias": "Vbias", "clk": "CLK", "clock": "CLK",
    "sig": "Sig", "signal": "Sig", "a": "A", "b": "B",
}


#: A bare identifier: a letter, then letters/digits/underscore. Short, because
#: a net name is written to be read off a wire, not a sentence.
#:
#: Letters first, then at most two trailing digits and nothing after them.
#: Letting a digit be followed by more letters is how ``RL1K`` -- the load
#: resistor's whole label, run together by the reader -- became a *net* called
#: RL1K: ``RL`` is not a designator this module recognises, so nothing else
#: rejected it. A net name does not carry a multiplier.
_GENERIC_NET_RE = re.compile(r"^[A-Za-z][A-Za-z_]{0,5}[0-9]{0,2}$")

#: Our own generic model names. They are not part numbers, so ``parse_label``
#: cannot reject them, and they are written on these drawings because this
#: project puts them there.
_NOT_NET_NAMES = {"DMOD", "QMOD", "MMOD", "SMOD", "OPAMP", "NPN", "PNP",
                  "NMOS", "PMOS", "AC", "DC"}


def parse_net_label(text: str | None) -> str | None:
    """Return the net name this string denotes, or ``None``.

    ``Vout`` next to a terminal names the wire. The same string next to a
    source would be a designator, which is why this is only ever consulted for
    terminal symbols -- the caller knows which it is looking at.
    """
    if not text:
        return None
    s = " ".join(text.translate(_SUBSCRIPTS).split())
    if _NET_LABEL_RE.match(s):
        return _NET_CANON.get(s.lower().replace(" ", ""), s)

    # Anything else short, beside a terminal, and not a part. The fixed list
    # above was written for the signals at the edge of a drawing -- Vin, Vout,
    # GND -- and a real schematic names its internal nets too: SW, FB, GATE,
    # OSC, CTRL, VLOAD. Rejecting those is not neutral, because the *point* of
    # writing a name on a wire is to join it to the other wire carrying the
    # same name. A boost converter drawn with its feedback marked FB at both
    # ends came back with the comparator input on a net of one pin, which
    # reads as a wire-tracing failure and is nothing of the kind: the drawing
    # said so in letters and we were not listening.
    #
    # The guard is that it must not be a *part*. ``R1`` beside a terminal is a
    # designator, ``2N2222`` is a part number, ``10k`` is a value, and
    # ``parse_label`` already recognises all three -- so whatever it cannot
    # explain, and that looks like an identifier, is a net name.
    if not _GENERIC_NET_RE.match(s) or s.upper() in _NOT_NET_NAMES:
        return None
    desig, value = parse_label(s)
    if desig is not None or value is not None:
        return None
    return s.upper()


def name_nets_from_labels(graph, detections: list, max_gap_frac: float = 3.0,
                          overrides: dict | None = None) -> int:
    """Name the nets that carry a label beside a terminal symbol.

    Without this every net is ``N001``..``N00n`` and a simulation can only be
    probed by guessing which number is the output. With it the netlist says
    ``Vout`` and ``.print V(Vout)`` works, which is the difference between a
    netlist that parses and one you can actually use.

    Two *names* for one net is a contradiction in the drawing, so the nearer
    label wins. Two *nets* with the same name is not a contradiction at all --
    it is the entire reason net labels exist. A drafter writes ``VCC`` at the
    top of two separate rails precisely so they need not draw a wire across
    the page, and wire tracing has no way to know that, because on the page
    there is nothing joining them.

    This used to skip the second one, leaving it as ``N006`` beside a ``VCC``
    it was supposed to be part of -- a circuit silently cut in half at exactly
    the point the drawing was being most explicit. Now the nets are merged,
    the same way :func:`topology.build_circuit` already fuses every ground
    symbol onto one node.

    Ground is never renamed: SPICE requires node ``0``. A label that reads
    ``GND`` on some other node is merged *into* ground for the same reason --
    a net called GND that is not node 0 is not a net name, it is a mistake.
    """
    terms = [d for d in detections if d.cls == "terminal"]
    pairs = []
    for t in detections:
        if t.cls != "text" or not getattr(t, "text", ""):
            continue
        name = parse_net_label(t.text)
        if not name:
            continue
        for term in terms:
            node = graph.terminal_nodes.get(id(term))
            if node is None or node == graph.ground_node:
                continue
            gap = _gap(t, term)
            if gap > max_gap_frac * max(term.w, term.h):
                continue
            # The *terminal's* id, not the node: merging below retires node
            # ids, and `terminal_nodes` is kept current through a merge, so
            # looking the node up again at use time is what stays correct.
            pairs.append((gap, id(t), id(term), name))

    pairs.sort(key=lambda p: p[0])
    used_text: set = set()
    n = 0
    for _gap_px, tid, term_id, name in pairs:
        if tid in used_text:
            continue
        node = graph.terminal_nodes.get(term_id)
        if node is None or node == graph.ground_node:
            used_text.add(tid)
            continue
        if node in graph.net_names:
            continue                      # this net already has a name
        held = next((k for k, v in graph.net_names.items()
                     if v.upper() == name.upper()), None)
        used_text.add(tid)
        if name.upper() == "GND":
            graph.merge_nodes(graph.ground_node, node)
        elif held is not None:
            graph.merge_nodes(held, node)
        else:
            graph.net_names[node] = name
            n += 1

    for node, name in (overrides or {}).items():
        if node != graph.ground_node:
            graph.net_names[int(node)] = name
    return n


def assign_labels(detections: list, max_gap_frac: float = 2.5) -> tuple[dict, dict]:
    """Match text boxes to components once, returning ``(values, names)``.

    One pass, not two. A box reading ``R1 1kΩ`` carries a designator and
    a value that belong to the *same* resistor, and matching them separately
    lets the two land on different ones -- the netlist then says R1 is 10 ohms
    while the drawing says 1k, with neither number wrong in isolation.

    Each text box is consumed once. A component may take a name from one box
    and a value from another, because people write ``C1`` above a capacitor and
    ``1pF`` below it, but it never takes two of either.
    """
    comps = [d for d in detections if d.cls in PLACEABLE]
    if not comps:
        return {}, {}

    pairs = []
    for t in detections:
        if t.cls != "text" or not getattr(t, "text", ""):
            continue
        desig, parsed = parse_label(t.text)
        if desig is None and parsed is None:
            continue
        # The designator's letter is the stronger constraint when present:
        # "C1" can only be a capacitor whatever the units say.
        allowed = (DESIGNATOR_CLASSES.get(desig[0].upper(), ()) if desig
                   else parsed.classes)
        for c in comps:
            if c.cls not in allowed:
                continue
            gap = _gap(t, c)
            if gap > max_gap_frac * max(c.w, c.h):
                continue
            pairs.append((gap, id(t), id(c), desig, parsed))

    pairs.sort(key=lambda p: p[0])
    used_text: set = set()
    values: dict = {}
    names: dict = {}
    claimed: set = set()
    for _gap_px, tid, cid, desig, parsed in pairs:
        if tid in used_text:
            continue
        want_name = desig is not None and cid not in names and desig.upper() not in claimed
        want_value = parsed is not None and cid not in values
        if not (want_name or want_value):
            continue
        used_text.add(tid)
        if want_name:
            names[cid] = desig
            claimed.add(desig.upper())
        if want_value:
            values[cid] = parsed.spice
    return values, names


def labels_from_detections(detections: list, max_gap_frac: float = 2.5) -> dict:
    """Designators only: ``{id(detection): "R1"}``. See :func:`assign_labels`."""
    return assign_labels(detections, max_gap_frac)[1]


def values_from_detections(detections: list, max_gap_frac: float = 2.5) -> dict:
    """Values only: ``{id(detection): spice_value}``. See :func:`assign_labels`.

    Each text box goes to the nearest component **of a compatible class**, and
    each component takes at most one value. Pairs are consumed in order of
    increasing distance, so an unambiguous short-range pair is settled before a
    doubtful long-range one competes for the same symbol.

    ``max_gap_frac`` caps how far a label may sit from its component, as a
    multiple of the component's own size. A value written at the far side of
    the page is more likely to belong to something we failed to detect.
    """
    return assign_labels(detections, max_gap_frac)[0]


def apply_names_to_graph(graph, names: dict) -> int:
    """Rename components to the designators written on the drawing.

    Our own refs come from position on the page, so our ``R1`` is the topmost
    resistor and almost never the drafter's ``R1``. The netlist is then
    unreadable next to the drawing it came from -- every value looks wrong
    even when every value is right.

    Renaming has to happen **as one batch**. The usual case is a permutation:
    our R1 is their R2 and our R2 is their R1, and renaming one at a time
    refuses both, because each wanted name is still held by the other. So the
    wanted names are reserved first, then any component that did not get one
    but is sitting on a reserved name is moved out of the way.

    A rename is refused if it would change the SPICE prefix -- LTspice
    dispatches on that first letter, so calling a capacitor ``R3`` would turn
    it into a resistor.
    """
    want: dict = {}
    claimed: set = set()
    for comp in graph.components:
        w = names.get(id(comp.det))
        if not w or w[0].upper() != comp.ref[0].upper() or w.upper() in claimed:
            continue
        want[id(comp)] = w
        claimed.add(w.upper())

    if not want:
        return 0

    def fresh(prefix: str, taken: set) -> str:
        i = 1
        while f"{prefix}{i}".upper() in taken:
            i += 1
        return f"{prefix}{i}"

    taken = set(claimed)
    for comp in graph.components:
        if id(comp) not in want and comp.ref.upper() not in taken:
            taken.add(comp.ref.upper())

    renamed = 0
    was: dict = {}
    for comp in graph.components:
        old = comp.ref
        if id(comp) in want:
            if comp.ref != want[id(comp)]:
                comp.ref = want[id(comp)]
                renamed += 1
        elif comp.ref.upper() in claimed:
            comp.ref = fresh(comp.ref[0], taken)   # displaced by a real name
            taken.add(comp.ref.upper())
        if comp.ref != old:
            was[old] = comp.ref

    # Warnings raised while the graph was built name the *positional* ref, and
    # this function has just moved most of those names onto other parts. Left
    # alone, "V3 has all terminals on one node" points at a healthy V3 while
    # the shorted part is now called V2 -- the warning is not merely useless,
    # it accuses the wrong component. Rewrite them to match what the netlist
    # will say.
    notes = getattr(graph, "warnings", None)
    if was and notes:
        pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in was) + r")\b")
        notes[:] = [pat.sub(lambda m: was[m.group(1)], w) for w in notes]
    return renamed


#: Classes whose default is a *quantity* someone would have written down. A
#: diode's ``DMOD`` is a stand-in for a part number we never expect to read,
#: so it is not a missing measurement; a capacitor's ``1u`` is.
_QUANTIFIED = ("resistor", "capacitor", "inductor",
               "source_voltage", "source_current")


def defaulted_components(graph, read_ids: set | None = None,
                         explicit: set | None = None) -> list[str]:
    """Parts still carrying their class default, in netlist order.

    A default is indistinguishable from a reading once it is in the netlist:
    ``C2 N003 0 1u`` says nothing about whether ``1u`` came off the page or
    out of :data:`classes.SPECS`. Two capacitors on a drawing that plainly
    said ``10u`` and ``10n`` silently took ``1u``, which put an oscillator at
    45 Hz instead of 4.5 kHz -- a wrong answer that looks exactly like a right
    one. Saying which parts were guessed costs one line and makes it visible.
    """
    want = {r.upper() for r in (explicit or ())}
    out = []
    for c in graph.components:
        if c.cls not in _QUANTIFIED:
            continue
        if read_ids is not None and id(c) in read_ids:
            continue          # the reader got this one off the page
        if c.ref.upper() in want:
            continue          # the user stated it
        out.append(c.ref)
    return out


def apply_to_graph(graph, assignment: dict, overrides: dict | None = None,
                   applied: set | None = None) -> int:
    """Write read values onto a built graph. Returns how many were applied.

    Explicit ``--value`` overrides win, but they are applied by the *caller*,
    at the end: a ref only means what the user typed once the designators have
    been taken off the page. Applying them here, against positional refs, put
    ``--value C2=10u`` onto whichever capacitor was second down the page --
    which the drawing then renamed C1, destroying its real 100u.

    ``applied`` collects ``id(comp)`` for every part given a value read from
    the drawing. That is the only way to tell a reading from a default
    afterwards: a 1k resistor the reader got right and one that fell back to
    ``SPECS`` are the same four characters in the netlist.
    """
    overrides = overrides or {}
    n = 0
    for comp in graph.components:
        if comp.ref in overrides:
            continue
        v = assignment.get(id(comp.det))
        if v:
            comp.value = v
            n += 1
            if applied is not None:
                applied.add(id(comp))
    return n


#: SPICE magnitude suffixes. ``meg`` must be tried before ``m``.
_SPICE_SUFFIX = (
    ("meg", 1e6), ("mil", 25.4e-6),
    ("t", 1e12), ("g", 1e9), ("k", 1e3),
    ("m", 1e-3), ("u", 1e-6), ("n", 1e-9), ("p", 1e-12), ("f", 1e-15),
)


def to_magnitude(text: str | None) -> float | str | None:
    """Canonicalise a SPICE value for comparison.

    Returns a float for a quantity, an upper-case string for a model name, and
    ``None`` for something with no value at all. This is what lets ``1µ`` from a
    reference netlist and ``1u`` from us compare equal, and ``10k`` equal
    ``10000``.

    It also has to survive mojibake: the reference ``.net`` files are read as
    latin-1, so a micro sign arrives as ``Âµ``. Stripping non-ASCII before
    parsing handles that without having to guess the file's encoding.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    # Drop the mojibake lead-in and any stray non-ASCII, mapping micro to 'u'.
    s = s.replace("Âµ", "u").replace("µ", "u").replace("μ", "u")
    s = "".join(ch for ch in s if ord(ch) < 128).strip()
    if not s:
        return None

    # LTspice hangs parameters off a value -- ``100K tol=1 pwr=0.1``, or a
    # whole tantalum part number on a capacitor. Only the leading magnitude is
    # comparable; the rest is component metadata a drawing never carries.
    head = s.split()
    if len(head) > 1 and any("=" in tok for tok in head[1:]):
        s = head[0]

    low = s.lower()
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-z]*)$", low)
    if not m:
        return s.upper()                       # a model name or a function
    num, suf = float(m.group(1)), m.group(2)
    if not suf:
        return num
    for name, scale in _SPICE_SUFFIX:
        if suf.startswith(name):
            return num * scale
    return num                                  # bare unit letter, e.g. "10v"


def values_equal(a, b, rel_tol: float = 1e-6) -> bool:
    """Compare two canonicalised values, numerically where both are numbers."""
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-30)
    return a == b


def read_text_boxes(image, detections: list, backend: str = "auto") -> int:
    """Fill in ``Detection.text`` for text boxes by running a recogniser.

    With ground-truth annotations the strings come from CGHD's ``<text>``
    elements and this is unnecessary; with the trained detector there is no
    such source, so without a recogniser every value falls back to its default.
    That is the one gap between the measured ceiling (perfect reading of the
    handwriting) and the full system.

    No recogniser is bundled: handwriting OCR is a project in itself, and the
    choice of engine is the user's. Install any one of the supported backends
    and value reading starts working with no other change:

        pip install pytesseract        # also needs the Tesseract binary
        pip install easyocr

    Returns the number of boxes that produced a string, so a caller can report
    "0 values read" rather than silently using defaults.
    """
    boxes = [d for d in detections if d.cls == "text" and not getattr(d, "text", "")]
    if not boxes:
        return 0

    reader = _ocr_reader(backend)
    if reader is None:
        return 0

    import numpy as np

    h, w = image.shape[:2]
    n = 0
    for d in boxes:
        # Margin, scaled to the *height* -- that is the character size, and a
        # box's width tells you nothing about how much of a word is missing.
        # Detector boxes clip horizontally: on one drawing "Vin" is boxed from
        # the i onward, so a reader can only ever return "in" however good it
        # is. Pad wider than tall, because that is the direction it fails in.
        px = max(3, int(0.35 * d.h))
        py = max(2, int(0.15 * d.h))
        x1, y1 = max(0, int(d.x1) - px), max(0, int(d.y1) - py)
        x2, y2 = min(w, int(d.x2) + px), min(h, int(d.y2) + py)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            txt = reader(np.ascontiguousarray(crop))
        except Exception:
            continue
        if txt:
            d.text = txt.strip()
            n += 1
    return n


def _ocr_reader(backend: str):
    """Return ``crop -> str``, or ``None`` when no backend is installed.

    Order matters, and it is not the obvious one. These labels are
    **handwriting**, and the general-purpose OCR engines are not:

    - ``easyocr`` was measured on this project's own drawings and returned
      ``Cour`` for ``Cout``, ``RITR`` for ``R1 1kΩ``, and nothing at all
      for half the boxes, at confidences around 0.3. Constraining the alphabet
      and binarising the crops made it worse, not better. Its recogniser is a
      CRNN trained on printed and scene text.
    - ``trocr`` (``microsoft/trocr-base-handwritten``) is trained on the IAM
      handwriting corpus, which is the task we actually have. It needs
      ``transformers`` and downloads ~1.4 GB on first use.

    So TrOCR is tried first and the others are fallbacks, not preferences.
    """
    if backend in ("auto", "trocr"):
        try:
            from PIL import Image
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            name = os.environ.get("CIRCUIT2SPICE_TROCR",
                                  "microsoft/trocr-base-handwritten")
            proc = TrOCRProcessor.from_pretrained(name)
            model = VisionEncoderDecoderModel.from_pretrained(name)
            model.eval()

            def run(crop):
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else                     cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                px = proc(images=Image.fromarray(rgb), return_tensors="pt").pixel_values
                ids = model.generate(px, max_new_tokens=16)
                return proc.batch_decode(ids, skip_special_tokens=True)[0]

            return run
        except Exception:
            if backend == "trocr":
                return None

    if backend in ("auto", "pytesseract"):
        try:
            import pytesseract

            def run(crop):
                # Restrict the alphabet: these labels are values and part
                # numbers, so letting the engine propose punctuation mostly
                # invents decimal points that were never written.
                cfg = ("--psm 7 -c tessedit_char_whitelist="
                       "0123456789.,kKMmuUnNpPGgRCLVIDQΩohmFfHhVvAaTIPNE")
                return pytesseract.image_to_string(crop, config=cfg)

            return run
        except Exception:
            if backend == "pytesseract":
                return None

    if backend in ("auto", "easyocr"):
        try:
            import easyocr

            rd = easyocr.Reader(["en"], gpu=False, verbose=False)

            def run(crop):
                res = rd.readtext(crop, detail=0)
                return " ".join(res) if res else ""

            return run
        except Exception:
            return None
    return None


def retag_rail_grounds(detections: list, max_gap_frac: float = 3.0) -> int:
    """Turn a ``gnd`` the drawing names something else into a ``terminal``.

    A rail marker -- a stub with one bar across the top -- and a ground
    symbol -- a stub with two or three -- are the same shape to within one
    stroke, and the detector duly returns ``gnd`` for both. Measured on one
    drawing: the VCC markers above a 12 V source and its decoupling capacitor
    came back as ``gnd`` at 0.58 and 0.46 with no competing ``terminal``, so
    :func:`topology._shadowed_grounds` had nothing to compare them against.
    Both parts ended up with their terminals on node 0 -- the supply shorted
    to its own return.

    The drawing settles it, in letters: **nobody labels a ground symbol
    VCC.** A ground carrying a net label that is not ground is a rail marker
    that was read wrong, and re-typing it is strictly safer than leaving it,
    because a false ground does not add a wrong node -- it merges every node
    that touches one.

    Conservative on purpose. It fires only on an explicit contradicting
    label, so an unlabelled ground, or one labelled ``GND``, is untouched.
    """
    grounds = [d for d in detections if d.cls == "gnd"]
    labels = [d for d in detections
              if d.cls == "text" and getattr(d, "text", "")]
    if not grounds or not labels:
        return 0
    n = 0
    for g in grounds:
        best, best_gap = None, None
        for t in labels:
            gap = _gap(t, g)
            if gap > max_gap_frac * max(g.w, g.h):
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = t, gap
        if best is None:
            continue
        name = parse_net_label(best.text)
        if name and name.upper() not in ("GND", "0"):
            g.cls = "terminal"
            n += 1
    return n
