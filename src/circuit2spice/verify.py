"""Round-trip verification of generated schematics.

Writing a ``.asc`` that *looks* right is easy; writing one whose wires actually
connect is not, and LTspice fails silently when they do not -- the schematic
opens, the component simply floats.

So we check our work against LTspice itself. We write the schematic, ask
``LTspice.exe -netlist`` what circuit it sees, and compare that against the
circuit we meant to describe.

The comparison is on the *partition of pins into nets*, not on node names:
LTspice numbers nets its own way (``N001``, ``N002``, ...), so requiring equal
names would be meaningless. Two pins must be on the same LTspice net exactly
when our graph says they share a node. Because LTspice preserves instance
names, pins can be matched between the two descriptions unambiguously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .asc_writer import write_asc
from .classes import spec
from .graph import CircuitGraph
from . import ltspice


@dataclass
class VerifyResult:
    ok: bool
    checked_pins: int = 0
    mismatches: list = field(default_factory=list)
    missing_refs: list = field(default_factory=list)
    extra_refs: list = field(default_factory=list)
    floating_pins: list = field(default_factory=list)
    netlist_text: str = ""

    def summary(self) -> str:
        if self.ok:
            return f"OK: {self.checked_pins} pins, net partition matches LTspice"
        bits = []
        if self.missing_refs:
            bits.append(f"{len(self.missing_refs)} components absent from netlist")
        if self.extra_refs:
            bits.append(f"{len(self.extra_refs)} unexpected components")
        if self.floating_pins:
            bits.append(f"{len(self.floating_pins)} pins LTspice sees as unconnected")
        if self.mismatches:
            bits.append(f"{len(self.mismatches)} pin-pairs wired wrongly")
        return "FAILED: " + "; ".join(bits)


def _partition(pairs: dict) -> dict:
    """Map each pin to a canonical group id, from pin -> net-name."""
    groups: dict = {}
    out: dict = {}
    for pin, net in pairs.items():
        out[pin] = groups.setdefault(net, len(groups))
    return out


def verify_graph(graph: CircuitGraph, asc_path: str | Path, **kw) -> VerifyResult:
    """Write ``graph`` to ``asc_path`` and check LTspice agrees with it.

    Extra keyword arguments go through to :func:`write_asc`. Passing
    ``analysis`` matters: this function *rewrites* the file, so without it a
    caller that had already written the schematic with, say, ``.tran`` gets it
    silently replaced by the default ``.op`` and then wonders why the
    simulation produced an operating point.
    """
    write_asc(graph, asc_path, **kw)
    text = ltspice.netlist(asc_path)
    actual = ltspice.parse_netlist(text)

    intended: dict = {}
    for comp in graph.components:
        for i, node in enumerate(graph.wired_nodes(comp)):
            if node >= 0:
                intended[(comp.ref, i)] = f"node{node}"

    missing = [c.ref for c in graph.components if c.ref not in actual]
    extra = [r for r in actual if r not in {c.ref for c in graph.components}]

    actual_pins: dict = {}
    floating: list = []
    for comp in graph.components:
        nets = actual.get(comp.ref)
        if nets is None:
            continue
        for i, node in enumerate(graph.wired_nodes(comp)):
            if node < 0 or i >= len(nets):
                continue
            net = nets[i]
            if net.upper().startswith("NC_"):
                floating.append((comp.ref, i))
            actual_pins[(comp.ref, i)] = net

    shared = sorted(set(intended) & set(actual_pins))
    pi, pa = _partition({k: intended[k] for k in shared}), _partition(
        {k: actual_pins[k] for k in shared}
    )

    mismatches: list = []
    for a_idx in range(len(shared)):
        for b_idx in range(a_idx + 1, len(shared)):
            a, b = shared[a_idx], shared[b_idx]
            if (pi[a] == pi[b]) != (pa[a] == pa[b]):
                mismatches.append(
                    (a, b, "should be joined" if pi[a] == pi[b] else "should be separate")
                )

    ok = not (missing or mismatches or floating)
    return VerifyResult(
        ok=ok,
        checked_pins=len(shared),
        mismatches=mismatches,
        missing_refs=missing,
        extra_refs=extra,
        floating_pins=floating,
        netlist_text=text,
    )
