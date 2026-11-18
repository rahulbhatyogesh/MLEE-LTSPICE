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

    def node_name(self, node: int) -> str:
        """SPICE node name.  Ground is always ``0``."""
        return "0" if node == self.ground_node else f"N{node:03d}"

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
