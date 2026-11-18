"""Resolve detected symbols plus a node map into a connected circuit graph.

For each detected component we look at the annulus just outside its bounding
box and ask which electrical nodes touch it. The ``n_ports`` strongest hits are
that component's terminals. The ring is grown adaptively, because a component
drawn with short wire stubs may need a wider search before its neighbours
appear, while a component in a dense area needs a tight one to avoid grabbing a
wire that merely passes nearby.
"""

from __future__ import annotations

import numpy as np

from .classes import PLACEABLE, spec
from .graph import CircuitGraph, Component, Detection, UnionFind
from .wires import NodeMap, _clip_rect, build_node_map


def _ring_label_stats(labels: np.ndarray, box, k: int) -> dict:
    """label -> (pixel_count, mean_x, mean_y) within the annulus outside box."""
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = _clip_rect((x1 - k, y1 - k, x2 + k, y2 + k), labels.shape)
    if ox2 <= ox1 or oy2 <= oy1:
        return {}
    patch = labels[oy1:oy2, ox1:ox2].copy()
    ix1, iy1, ix2, iy2 = _clip_rect(box, labels.shape)
    patch[
        max(0, iy1 - oy1) : max(0, iy2 - oy1),
        max(0, ix1 - ox1) : max(0, ix2 - ox1),
    ] = 0
    ys, xs = np.nonzero(patch)
    if len(xs) == 0:
        return {}
    vals = patch[ys, xs]
    out: dict = {}
    for lab in np.unique(vals):
        sel = vals == lab
        out[int(lab)] = (
            int(sel.sum()),
            float(xs[sel].mean() + ox1),
            float(ys[sel].mean() + oy1),
        )
    return out


def _terminals(node_map: NodeMap, det: Detection, n_ports: int) -> list[tuple]:
    """Find up to ``n_ports`` (label, cx, cy) terminals, growing the ring."""
    base = max(4.0, 0.10 * max(det.w, det.h))
    best: list[tuple] = []
    for mult in (1.0, 1.8, 3.0, 4.5):
        k = int(round(base * mult))
        stats = _ring_label_stats(node_map.labels, det.box, k)
        # Ignore grazing contacts: a real stub leaves several skeleton pixels.
        hits = [(lab, c, cx, cy) for lab, (c, cx, cy) in stats.items() if c >= 3]
        hits.sort(key=lambda t: -t[1])
        if len(hits) >= len(best):
            best = [(lab, cx, cy) for lab, _c, cx, cy in hits[:n_ports]]
        if len(best) >= n_ports:
            break
    return best


def _orientation(det: Detection, terms: list[tuple]) -> str:
    """Pick R0/R90 from where the terminals sit relative to the body.

    LTspice symbols are drawn vertically, so a component whose terminals are
    left and right of its body is R90. Polarity (R0 vs R180) cannot be read
    from an axis-aligned box alone -- CGHD ships rotation annotations that
    would settle it, which is noted as future work.
    """
    if len(terms) < 2:
        return "R0" if det.h >= det.w else "R90"
    (_, x_a, y_a), (_, x_b, y_b) = terms[0], terms[1]
    return "R90" if abs(x_a - x_b) > abs(y_a - y_b) else "R0"


def build_circuit(
    bgr: np.ndarray,
    detections: list,
    *,
    node_map: NodeMap | None = None,
    values: dict | None = None,
    bridge_px: int | None = None,
    method: str = "flatfield",
    drop_outside: str = "none",
    thin_q: float | None = 50.0,
) -> tuple[CircuitGraph, NodeMap]:
    """Turn detections into a :class:`CircuitGraph`."""
    if node_map is None:
        node_map = build_node_map(bgr, detections, bridge_px=bridge_px, method=method,
                                  drop_outside=drop_outside, thin_q=thin_q)

    warnings: list[str] = []
    uf = UnionFind()
    for n in range(1, node_map.n_nodes + 1):
        uf.find(n)

    # Every ground symbol names the same electrical node, so tie them together.
    ground_labels: list[int] = []
    for d in detections:
        if d.cls != "gnd":
            continue
        terms = _terminals(node_map, d, 1)
        if terms:
            ground_labels.append(terms[0][0])
    for other in ground_labels[1:]:
        uf.union(ground_labels[0], other)

    # Resolve each placeable component to its terminals.
    raw: list[tuple] = []
    counters: dict = {}
    for d in sorted(detections, key=lambda d: (d.cy, d.cx)):
        if d.cls not in PLACEABLE:
            continue
        sp = spec(d.cls)
        terms = _terminals(node_map, d, sp.n_ports)
        if len(terms) < sp.n_ports:
            warnings.append(
                f"{d.cls} at ({d.cx:.0f},{d.cy:.0f}) found only "
                f"{len(terms)}/{sp.n_ports} terminals; left unconnected"
            )
        counters[sp.spice_prefix] = counters.get(sp.spice_prefix, 0) + 1
        ref = f"{sp.spice_prefix}{counters[sp.spice_prefix]}"
        raw.append((ref, d, sp, terms))

    # Decide which node is ground, then renumber so ground is 0.
    if ground_labels:
        ground_root = uf.find(ground_labels[0])
    else:
        degree: dict = {}
        for _ref, _d, _sp, terms in raw:
            for lab, _cx, _cy in terms:
                r = uf.find(lab)
                degree[r] = degree.get(r, 0) + 1
        if degree:
            ground_root = max(degree, key=lambda k: degree[k])
            warnings.append(
                "no ground symbol detected; using the highest-degree node "
                "as the reference node"
            )
        else:
            ground_root = -1
            warnings.append("no nodes found at all")

    remap: dict = {ground_root: 0}
    next_id = 1

    def node_id(label: int) -> int:
        nonlocal next_id
        root = uf.find(label)
        if root not in remap:
            remap[root] = next_id
            next_id += 1
        return remap[root]

    components: list[Component] = []
    for ref, d, sp, terms in raw:
        nodes = [node_id(lab) for lab, _cx, _cy in terms]
        while len(nodes) < sp.n_ports:
            nodes.append(-1)
        if len(set(n for n in nodes if n >= 0)) == 1 and sp.n_ports > 1:
            warnings.append(f"{ref} has all terminals on one node (short circuit)")
        value = (values or {}).get(ref, sp.default_value or "")
        components.append(
            Component(
                ref=ref,
                cls=d.cls,
                det=d,
                nodes=nodes,
                value=value,
                orientation=_orientation(d, terms),
            )
        )

    h, w = bgr.shape[:2]
    graph = CircuitGraph(
        components=components,
        n_nodes=next_id,
        ground_node=0,
        image_size=(w, h),
        warnings=warnings,
    )
    return graph, node_map
