"""Topology metrics for the extracted circuits.

CGHD gives bounding boxes but not, for most images, a ground-truth netlist. So
we score the extracted topology with three structural properties that any
correct circuit has, and which fail in *opposite* directions -- which is what
makes them useful together rather than individually:

``completeness``
    Fraction of component ports that were assigned an electrical node. Falls
    when wire tracing is too conservative and a stub is missed, and also when
    it is too aggressive and a component's two ports collapse onto one node.

``short_rate``
    Fraction of multi-port components whose ports all landed on the *same*
    node. A real circuit has almost none. Rises when node merging is too
    aggressive.

``connected_frac``
    Fraction of components lying in the largest connected sub-circuit. A real
    schematic is one connected circuit. Falls when node merging is too timid
    and a single net fragments, orphaning components.

Over-merging drives ``short_rate`` up; under-merging drives ``connected_frac``
down. The operating point that maximises ``score`` sits between them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .classes import spec
from .graph import CircuitGraph, UnionFind


@dataclass
class TopologyMetrics:
    images: int = 0
    components: int = 0
    ports: int = 0
    ports_resolved: int = 0
    shorted: int = 0
    multiport: int = 0
    in_largest: int = 0
    nodes: int = 0

    @property
    def completeness(self) -> float:
        return self.ports_resolved / self.ports if self.ports else 0.0

    @property
    def short_rate(self) -> float:
        return self.shorted / self.multiport if self.multiport else 0.0

    @property
    def connected_frac(self) -> float:
        return self.in_largest / self.components if self.components else 0.0

    @property
    def score(self) -> float:
        """Single number for sweeps: reward completeness and connectivity,
        penalise shorts."""
        return (self.completeness + self.connected_frac) / 2 - self.short_rate

    def as_row(self) -> dict:
        return {
            "images": self.images,
            "components": self.components,
            "completeness": round(self.completeness, 4),
            "short_rate": round(self.short_rate, 4),
            "connected_frac": round(self.connected_frac, 4),
            "score": round(self.score, 4),
            "mean_nodes": round(self.nodes / self.images, 2) if self.images else 0,
        }


def score_graph(graph: CircuitGraph, acc: TopologyMetrics | None = None) -> TopologyMetrics:
    m = acc or TopologyMetrics()
    m.images += 1
    m.nodes += graph.n_nodes

    uf = UnionFind()
    members: dict = {}
    for c in graph.components:
        m.components += 1
        n_ports = spec(c.cls).n_ports
        m.ports += n_ports
        resolved = [n for n in c.nodes if n >= 0]
        m.ports_resolved += len(resolved)
        if n_ports > 1:
            m.multiport += 1
            if resolved and len(set(resolved)) == 1:
                m.shorted += 1
        # Union the component with every node it touches, so components that
        # share a net end up in the same group.
        key = ("c", c.ref)
        uf.find(key)
        members[key] = c.ref
        for n in resolved:
            # Ground is shared by everything, so it must not be the thing that
            # makes the circuit look connected.
            if n == graph.ground_node:
                continue
            uf.union(key, ("n", n))

    if graph.components:
        sizes: dict = {}
        for key in members:
            root = uf.find(key)
            sizes[root] = sizes.get(root, 0) + 1
        m.in_largest += max(sizes.values()) if sizes else 0
    return m
