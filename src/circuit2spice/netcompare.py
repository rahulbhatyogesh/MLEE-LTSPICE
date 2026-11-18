"""Compare an extracted circuit against a reference netlist.

CGHD ships twelve reference LTspice schematics for drafter 1 -- one per circuit,
and each circuit was drawn twice and photographed four times, so those twelve
files are ground truth for ninety-six photographs. That turns topology
extraction from something we can only inspect by eye into something we can
score.

A circuit is represented as a bipartite graph: component nodes labelled by SPICE
element type (``R``, ``C``, ``Q``, ...) joined to net nodes. Comparing circuits
is then graph isomorphism, with ground pinned to ground so the match cannot
permute the reference node onto a signal net. Node *names* are deliberately
ignored -- LTspice numbers nets its own way and our numbering comes from
connected-component order, so any name-based comparison would measure nothing.

Exact isomorphism is strict and all-or-nothing, so we also report a per-component
F1 over local signatures, which gives partial credit when most of a circuit is
right and one terminal is misplaced.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx

GROUND_NAMES = {"0", "gnd", "GND"}


def netlist_to_graph(elements: dict, pin_ground: bool = True) -> nx.Graph:
    """Bipartite component/net graph from ``{ref: [net, ...]}``."""
    g = nx.Graph()
    for ref, nets in elements.items():
        etype = ref[0].upper()
        cnode = ("c", ref)
        g.add_node(cnode, kind="component", label=etype)
        for i, net in enumerate(nets):
            is_gnd = net in GROUND_NAMES
            nnode = ("n", "0" if is_gnd else net)
            if nnode not in g:
                label = "gnd" if (is_gnd and pin_ground) else "net"
                g.add_node(nnode, kind="net", label=label)
            g.add_edge(cnode, nnode)
    return g


def _node_match(a: dict, b: dict) -> bool:
    return a.get("label") == b.get("label")


def is_isomorphic(a: nx.Graph, b: nx.Graph) -> bool:
    return nx.is_isomorphic(a, b, node_match=_node_match)


def _signatures(g: nx.Graph) -> Counter:
    """Local topology signature per component: its own type plus, for each of
    its nets, that net's sorted multiset of attached component types."""
    sigs: Counter = Counter()
    for node, data in g.nodes(data=True):
        if data.get("kind") != "component":
            continue
        legs = []
        for net in g.neighbors(node):
            peers = sorted(
                g.nodes[p]["label"] for p in g.neighbors(net) if p != node
            )
            legs.append((g.nodes[net]["label"], tuple(peers)))
        sigs[(data["label"], tuple(sorted(legs)))] += 1
    return sigs


def signature_f1(pred: nx.Graph, ref: nx.Graph) -> tuple[float, float, float]:
    sp, sr = _signatures(pred), _signatures(ref)
    tp = sum((sp & sr).values())
    p = tp / max(1, sum(sp.values()))
    r = tp / max(1, sum(sr.values()))
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def element_counts(elements: dict) -> Counter:
    return Counter(ref[0].upper() for ref in elements)


def count_f1(pred: dict, ref: dict) -> tuple[float, float, float]:
    """Precision/recall/F1 on the multiset of element types (a bill of materials
    score, comparable to what the reference papers report)."""
    cp, cr = element_counts(pred), element_counts(ref)
    tp = sum((cp & cr).values())
    p = tp / max(1, sum(cp.values()))
    r = tp / max(1, sum(cr.values()))
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class Comparison:
    stem: str
    circuit: str
    exact: bool
    bom_p: float
    bom_r: float
    bom_f1: float
    sig_p: float
    sig_r: float
    sig_f1: float
    n_pred: int
    n_ref: int

    def as_row(self) -> dict:
        return {
            "stem": self.stem, "circuit": self.circuit, "exact": self.exact,
            "bom_f1": round(self.bom_f1, 4), "sig_f1": round(self.sig_f1, 4),
            "n_pred": self.n_pred, "n_ref": self.n_ref,
        }


def compare(stem: str, circuit: str, pred: dict, ref: dict,
            pin_ground: bool = True) -> Comparison:
    """``pin_ground=False`` when the drawing shows no ground symbol: the
    reference schematic still has to pick *some* reference node, but that choice
    is not recoverable from the drawing, so requiring it to match would penalise
    a topologically correct extraction."""
    gp = netlist_to_graph(pred, pin_ground)
    gr = netlist_to_graph(ref, pin_ground)
    bp, br, bf = count_f1(pred, ref)
    sp, sr_, sf = signature_f1(gp, gr)
    return Comparison(
        stem=stem, circuit=circuit,
        exact=is_isomorphic(gp, gr),
        bom_p=bp, bom_r=br, bom_f1=bf,
        sig_p=sp, sig_r=sr_, sig_f1=sf,
        n_pred=len(pred), n_ref=len(ref),
    )
