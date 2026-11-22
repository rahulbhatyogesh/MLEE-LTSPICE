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


def _is_multi(name: str, nets: list) -> bool:
    return len(nets) > 2


def _is_diode(name: str, nets: list) -> bool:
    return len(nets) == 2 and name[:1].upper() == "D"


def pin_role_accuracy(pred: dict, ref: dict, pin_ground: bool = True,
                      max_maps: int = 2000, select=_is_multi) -> tuple[int, int]:
    """``(correct, total)`` pin orders for the devices ``select`` picks out.

    Exact match cannot see this. :func:`netlist_to_graph` is undirected and
    :func:`_signatures` sorts each component's legs, so a transistor with its
    collector and emitter transposed is isomorphic to the correct circuit --
    which is right for the question "is the wiring the same" and wrong for
    "will it simulate". This asks the second question.

    Net names do not correspond between the two netlists, so the mapping comes
    from the isomorphism itself. Several isomorphisms usually exist (a pair of
    identical parallel resistors makes two), and they are not equally kind to
    us, so we score the best one -- scoring an arbitrary one would measure
    which mapping the search happened to emit first. Returns ``(0, 0)`` when
    the graphs are not isomorphic at all, since without a correspondence
    between nets there is nothing to compare.
    """
    multi = {r: n for r, n in ref.items() if select(r, n)}
    if not multi:
        return 0, 0
    gp = netlist_to_graph(pred, pin_ground)
    gr = netlist_to_graph(ref, pin_ground)
    gm = nx.algorithms.isomorphism.GraphMatcher(gp, gr, node_match=_node_match)

    best = -1
    total = sum(1 for r, n in pred.items() if select(r, n))
    if total == 0:
        return 0, len(multi)
    for seen, mapping in enumerate(gm.isomorphisms_iter()):
        if seen >= max_maps:
            break
        correct = 0
        for p_ref, p_nets in pred.items():
            if not select(p_ref, p_nets):
                continue
            r_node = mapping.get(("c", p_ref))
            if r_node is None:
                continue
            r_nets = ref.get(r_node[1])
            if r_nets is None or len(r_nets) != len(p_nets):
                continue
            # Map each of our nets into the reference's naming and compare
            # pin by pin: same nets in the same order, or it does not count.
            ours = []
            for net in p_nets:
                n_node = mapping.get(("n", "0" if net in GROUND_NAMES else net))
                ours.append(n_node[1] if n_node else None)
            theirs = ["0" if n in GROUND_NAMES else n for n in r_nets]
            correct += int(ours == theirs)
        best = max(best, correct)
        if best == total:
            break
    return (max(best, 0), total) if best >= 0 else (0, 0)


def polarity_accuracy(pred: dict, ref: dict, pin_ground: bool = True,
                      max_maps: int = 2000) -> tuple[int, int]:
    """``(correct, total)`` for the two-terminal *polar* parts: the diodes.

    The same blind spot as :func:`pin_role_accuracy`, one terminal fewer.
    Exact match sorts a component's legs before comparing, so a diode soldered
    in backwards is isomorphic to one the right way round -- the wiring is the
    same and the circuit is not, because a reversed diode conducts when it
    should block. This asks the question exact match throws away.
    """
    return pin_role_accuracy(pred, ref, pin_ground, max_maps, select=_is_diode)


def value_f1(pred: dict, ref: dict) -> tuple[float, float, float]:
    """Precision/recall/F1 over the multiset of (element type, value) pairs.

    A *bill of values*, deliberately built the same way as :func:`count_f1`'s
    bill of materials: a multiset comparison needs no correspondence between
    our instance names and LTspice's, which do not line up because ours come
    from position on the page and theirs from the reference schematic.

    Values are compared numerically, so ``10k`` matches ``10000`` and the
    reference's ``1µ`` matches our ``1u``.
    """
    from .values import to_magnitude

    def bag(d: dict) -> Counter:
        c: Counter = Counter()
        for r, v in d.items():
            mag = to_magnitude(v)
            if isinstance(mag, float):
                mag = round(mag, 12)
            c[(r[0].upper(), mag)] += 1
        return c

    bp, br = bag(pred), bag(ref)
    tp = sum((bp & br).values())
    p = tp / max(1, sum(bp.values()))
    r = tp / max(1, sum(br.values()))
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


#: Reference "values" that carry no information: LTspice prints the bare symbol
#: letter when a component's Value attribute is empty, and a generic model name
#: when none was chosen.
_UNVALUED = {"R", "C", "L", "V", "I", "D", "Q", "M", "S", "NPN", "PNP", "NMOS", "PMOS"}


def informative_values(ref: dict) -> dict:
    """The reference entries that actually state a value.

    Several CGHD reference schematics were drawn without filling in any
    component values -- C10's netlist is literally ``R1 R`` / ``C1 C`` -- and
    some sources carry an AC spec (``AC 2``, ``SINE(0 20m 400)``) that the
    photograph never shows. Scoring against those measures the reference's
    completeness, not our reading, so they are reported separately.
    """
    out = {}
    for r, v in ref.items():
        s = (v or "").strip()
        if not s or s.upper() in _UNVALUED:
            continue
        if "(" in s or s.upper().startswith(("AC ", "DC ", "PULSE", "SINE")):
            continue
        out[r] = v
    return out
