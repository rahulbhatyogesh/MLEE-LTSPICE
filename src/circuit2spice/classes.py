"""CGHD label taxonomy, merged down to the classes LTspice can actually place.

CGHD (Bayer et al.) ships 59 fine-grained labels.  Many of them are variants that
map onto the same LTspice symbol (``resistor``, ``resistor.adjustable`` and
``resistor.photo`` are all ``res``), and several are annotation artefacts rather
than components.  Training on all 59 spreads a small dataset across a long tail
that the downstream netlist stage cannot use anyway, so we merge to 15 classes.

The merge is done by *prefix*, so unseen CGHD sub-variants (the dataset is still
being extended upstream) fall into the right bucket automatically instead of
silently becoming a new class.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Merged taxonomy
# --------------------------------------------------------------------------

#: Ordered list of merged class names.  Index == YOLO class id; this order is
#: baked into the trained weights, so do not reorder without retraining.
NAMES: list[str] = [
    "resistor",
    "capacitor",
    "inductor",
    "diode",
    "transistor_bjt",
    "transistor_fet",
    "source_voltage",
    "source_current",
    "opamp",
    "gnd",
    "junction",
    "crossover",
    "switch",
    "terminal",
    "text",
    # Catch-all for symbols we cannot turn into a SPICE element: ICs, logic
    # gates, motors, speakers, probes, explanatory scribbles.  These are NOT
    # dropped, because a symbol the detector cannot see is a symbol we cannot
    # erase -- and un-erased ink survives wire tracing as a phantom node that
    # shorts together everything it touches.  Detecting them lets us remove
    # them cleanly and report the circuit as partially unsupported.
    "other_component",
]

NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(NAMES)}

#: (CGHD label prefix, merged name).  Order matters: the first prefix that
#: matches wins, so longer/more specific prefixes come first.
_MERGE_RULES: tuple[tuple[str, str], ...] = (
    ("transistor.bjt", "transistor_bjt"),
    ("transistor.fet", "transistor_fet"),
    ("transistor.mos", "transistor_fet"),
    ("transistor.photo", "transistor_bjt"),
    ("transistor", "transistor_bjt"),
    ("resistor", "resistor"),
    ("capacitor", "capacitor"),
    ("inductor", "inductor"),
    ("diode", "diode"),
    ("operational_amplifier", "opamp"),
    ("voltage", "source_voltage"),
    ("current", "source_current"),
    ("battery", "source_voltage"),
    ("gnd", "gnd"),
    ("ground", "gnd"),
    ("junction", "junction"),
    ("crossover", "crossover"),
    ("switch", "switch"),
    ("terminal", "terminal"),
    ("connector", "terminal"),
    ("port", "terminal"),
    ("text", "text"),
)


def merge_label(raw: str | None) -> str | None:
    """Map a raw CGHD label onto a merged class name.

    Anything with no specific rule becomes ``other_component`` rather than
    being discarded: see the note on that class in :data:`NAMES`.
    """
    if not raw:
        return None
    name = raw.strip().lower().replace(" ", "_").replace("-", "_")
    for prefix, merged in _MERGE_RULES:
        if name.startswith(prefix):
            return merged
    return "other_component"


# --------------------------------------------------------------------------
# Electrical / LTspice properties of each merged class
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    """How a merged class behaves electrically and in LTspice."""

    name: str
    n_ports: int
    #: SPICE instance-name prefix (``R1``, ``C2``, ...).  ``None`` => not a
    #: circuit element (junction, text, ...).
    spice_prefix: str | None
    #: LTspice ``.asy`` symbol name, or ``None`` if the class is not placed as
    #: a symbol (ground becomes a ``FLAG``, junctions become node merges).
    ltspice_symbol: str | None
    #: Default value used when no handwritten value is recognised.
    default_value: str | None = None
    #: True for classes that only affect connectivity, never the netlist.
    is_connectivity: bool = False


SPECS: dict[str, ComponentSpec] = {
    "resistor":       ComponentSpec("resistor",       2, "R",  "res",            "1k"),
    "capacitor":      ComponentSpec("capacitor",      2, "C",  "cap",            "1u"),
    "inductor":       ComponentSpec("inductor",       2, "L",  "ind",            "1m"),
    "diode":          ComponentSpec("diode",          2, "D",  "diode",          "DMOD"),
    "transistor_bjt": ComponentSpec("transistor_bjt", 3, "Q",  "npn",            "QMOD"),
    "transistor_fet": ComponentSpec("transistor_fet", 3, "M",  "nmos",           "MMOD"),
    "source_voltage": ComponentSpec("source_voltage", 2, "V",  "voltage",        "5"),
    "source_current": ComponentSpec("source_current", 2, "I",  "current",        "1m"),
    # 3-pin ideal op-amp: a hand-drawn triangle almost never shows supply
    # rails, and UniversalOpAmp2 would leave V+/V- dangling.
    "opamp":          ComponentSpec("opamp",          3, "X",  "OpAmps\\opamp",  "opamp"),
    "switch":         ComponentSpec("switch",         2, "S",  "sw",             "SMOD"),
    "gnd":            ComponentSpec("gnd",            1, None, None,  is_connectivity=True),
    "junction":       ComponentSpec("junction",       0, None, None,  is_connectivity=True),
    "crossover":      ComponentSpec("crossover",      0, None, None,  is_connectivity=True),
    "terminal":       ComponentSpec("terminal",       1, None, None,  is_connectivity=True),
    "text":           ComponentSpec("text",           0, None, None,  is_connectivity=True),
    "other_component": ComponentSpec("other_component", 0, None, None, is_connectivity=True),
}

#: Classes that become actual netlist elements.
PLACEABLE: frozenset[str] = frozenset(
    n for n, s in SPECS.items() if s.spice_prefix is not None
)

#: Classes whose pixels must be erased before wire tracing, but which are not
#: themselves circuit elements.
ERASE_BEFORE_WIRES: frozenset[str] = frozenset(
    PLACEABLE | {"text", "gnd", "terminal", "other_component"}
)


#: ``.model`` cards emitted for the generic model names used as defaults, so a
#: generated netlist simulates standalone instead of erroring on a missing
#: model.  Keyed by the default value string used in :data:`SPECS`.
MODEL_CARDS: dict[str, str] = {
    "DMOD": ".model DMOD D(Is=1e-14 Rs=0.1 N=1)",
    "QMOD": ".model QMOD NPN(Bf=100 Is=1e-15)",
    "MMOD": ".model MMOD NMOS(Vto=1 Kp=2e-5)",
    "SMOD": ".model SMOD SW(Ron=1 Roff=1Meg Vt=0.5)",
}

#: Extra ``.lib`` includes required by particular LTspice symbols.
SYMBOL_LIBS: dict[str, str] = {"OpAmps\\opamp": "opamp.sub"}


def spec(name: str) -> ComponentSpec:
    return SPECS[name]


def n_ports(name: str) -> int:
    return SPECS[name].n_ports if name in SPECS else 2
