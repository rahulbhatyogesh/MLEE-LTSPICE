"""Core data structures shared by the detection, topology and export stages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .classes import spec


@dataclass
class Detection:
    """One detected symbol, in image pixel coordinates."""

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    #: What a handwritten label says, when it is known. Filled from CGHD's
    #: ``<text>`` annotation, or by a recogniser; empty otherwise.
    text: str = ""
    #: What the reader actually returned, before `repair_reading` was applied
    #: under the neighbouring component's unit. Kept because that repair is
    #: lossy and unit-dependent: beside a resistor, "100u" is repaired to
    #: "100ohm" (a sub-microohm resistor being implausible), and if the part
    #: turns out to be an inductor the microhenries cannot be recovered from
    #: the repaired string. See `ocr.rerepair`.
    text_raw: str = ""

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def box(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def dilated(self, k: float) -> tuple[float, float, float, float]:
        return self.x1 - k, self.y1 - k, self.x2 + k, self.y2 + k


@dataclass
class Component:
    """A detected symbol that has been resolved to electrical nodes."""

    ref: str                      # SPICE instance name, e.g. "R1"
    cls: str                      # merged class name
    det: Detection
    nodes: list[int]              # electrical node id per port, in port order
    value: str
    orientation: str = "R0"       # R0 / R90 / R180 / R270
    #: Nodes for pins the *symbol* has but the drawing does not show, and
    #: which therefore cannot be traced: a SPICE switch's two control pins.
    #: ``None`` means tie them to ground, which leaves the switch open.
    control: list | None = None

    @property
    def is_complete(self) -> bool:
        """True when every port found a distinct-enough node."""
        return len(self.nodes) == spec(self.cls).n_ports and all(
            n >= 0 for n in self.nodes
        )


@dataclass
class CircuitGraph:
    """The extracted circuit: components plus the node set they connect."""

    components: list[Component] = field(default_factory=list)
    n_nodes: int = 0
    ground_node: int = 0
    image_size: tuple[int, int] = (0, 0)
    #: Non-fatal problems worth surfacing to the user and the report.
    warnings: list[str] = field(default_factory=list)
    #: node id -> the name written beside it on the drawing (``Vout``).
    net_names: dict = field(default_factory=dict)
    #: id(detection) -> node id, for terminal symbols. Lets a label read after
    #: the graph was built still find the net it belongs to.
    terminal_nodes: dict = field(default_factory=dict)

    def wired_nodes(self, comp) -> list:
        """One node per *symbol* pin, which is not one per modelled port.

        A hand-drawn switch has two terminals; LTspice's ``sw`` has four, the
        extra pair being the control input that decides whether it is closed.
        Nothing in the drawing traces to those pins, so they are not ports and
        ``_terminals`` must not go looking for them -- but they do have to be
        written, and until now ``netlist.py`` hard-coded them to ``0 0`` while
        ``asc_writer`` left them off the schematic entirely. The two outputs
        disagreed about the same circuit, and the switch was permanently open
        in both.
        """
        if comp.cls == "switch":
            ctrl = comp.control or [self.ground_node, self.ground_node]
            return list(comp.nodes) + [int(n) for n in ctrl]
        return list(comp.nodes)

    def merge_nodes(self, keep: int, drop: int) -> None:
        """Make ``drop`` the same electrical node as ``keep``.

        Used when the drawing says two separated wires are one net -- the
        whole purpose of writing ``VCC`` at both ends rather than drawing a
        line across the page. Ground symbols are already fused this way in
        :func:`topology.build_circuit`; a net *label* is the same statement
        made in text.

        The dropped id is retired rather than reused, and ``n_nodes`` stays
        put: it counts the ids ever handed out, and renumbering here would
        invalidate every node number the caller has already been shown --
        including the ones ``--net`` takes.
        """
        if keep == drop:
            return
        for c in self.components:
            c.nodes = [keep if n == drop else n for n in c.nodes]
            if c.control:
                c.control = [keep if n == drop else n for n in c.control]
        self.net_names.pop(drop, None)
        for k, v in list(self.terminal_nodes.items()):
            if v == drop:
                self.terminal_nodes[k] = keep

    def node_name(self, node: int) -> str:
        """SPICE node name. Ground is always ``0``.

        A net the drafter named -- ``Vin``, ``Vout`` -- keeps that name.
        Everything downstream goes through here, so the netlist, the LTspice
        ``FLAG``s and the JSON all agree, and a simulation can be probed as
        ``V(Vout)`` instead of ``V(N004)``.
        """
        if node == self.ground_node:
            return "0"
        return self.net_names.get(node) or f"N{node:03d}"

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "ground_node": self.ground_node,
            "image_size": list(self.image_size),
            "warnings": self.warnings,
            "components": [
                {
                    "ref": c.ref,
                    "cls": c.cls,
                    "value": c.value,
                    "orientation": c.orientation,
                    "nodes": [self.node_name(n) for n in c.nodes],
                    "bbox": [c.det.x1, c.det.y1, c.det.x2, c.det.y2],
                    "conf": c.det.conf,
                }
                for c in self.components
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class UnionFind:
    """Tiny disjoint-set, used to merge wire fragments into electrical nodes."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self._parent.setdefault(x, x)
        while p != x:
            self._parent[x] = x = self._parent.setdefault(p, p)
            p = self._parent.setdefault(x, x)
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for x in list(self._parent):
            out.setdefault(self.find(x), []).append(x)
        return out
