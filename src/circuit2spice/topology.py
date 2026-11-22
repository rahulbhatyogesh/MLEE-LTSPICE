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

from .classes import PLACEABLE, SPECS, spec
from .graph import CircuitGraph, Component, Detection, UnionFind
from .diode_net import order_diode_terminals
from .opamp import order_opamp_terminals
from .pins import (
    order_fet_terminals, order_source_terminals, order_transistor_terminals,
)
from .wires import NodeMap, _ring_label_stats, build_node_map


def reclassify(graph: CircuitGraph, overrides: dict) -> None:
    """Change a part's class by hand, keeping the wiring the drawing gave.

    This exists because some symbols are not *learnable* from CGHD, which is
    the only training data there is. The sharpest case: CGHD contains **1314
    voltage sources and zero current sources**, so a detector trained on it
    has never seen one and has no competing hypothesis to offer -- it labels a
    current source ``source_voltage`` at 0.91 confidence and is not going to
    stop doing so for any amount of retraining on this dataset.

    An override is therefore the correct answer rather than a workaround, and
    it is cheap: the terminals were found from the *ink*, which does not care
    what the symbol turned out to be, so only the class, the SPICE prefix and
    the default value change. The port count must match, because a class with
    more pins would need terminals nothing has looked for.
    """
    if not overrides:
        return
    want = {k.upper(): v for k, v in overrides.items()}
    for comp in graph.components:
        new = want.pop(comp.ref.upper(), None)
        if new is None:
            continue
        if new not in SPECS:
            graph.warnings.append(f"unknown class {new!r} for {comp.ref}")
            continue
        old, sp = spec(comp.cls), spec(new)
        if sp.n_ports != old.n_ports:
            graph.warnings.append(
                f"cannot re-type {comp.ref} as {new}: {sp.n_ports} ports "
                f"against {old.n_ports} found in the drawing")
            continue
        if sp.spice_prefix is None:
            graph.warnings.append(f"{new} is not a SPICE element")
            continue
        taken = {c.ref.upper() for c in graph.components if c is not comp}
        i = 1
        while f"{sp.spice_prefix}{i}".upper() in taken:
            i += 1
        was = comp.ref
        comp.ref = f"{sp.spice_prefix}{i}"
        # Only a value that is still the old class's default is replaced; one
        # read off the drawing is a measurement and survives re-typing.
        if comp.value == (old.default_value or ""):
            comp.value = sp.default_value or ""
        comp.cls = new
        # The *detection* is re-typed too, not just the component. Label
        # matching keys off the detection's class -- "L1" is only offered to
        # an inductor -- so without this an inductor the detector called a
        # resistor can never take its own designator or its own value off the
        # page, and comes back as some other L with somebody else's henries.
        comp.det.cls = new
        graph.warnings.append(f"re-typed {was} as {new} ({comp.ref}) by hand")
    for ref in want:
        graph.warnings.append(f"no component called {ref} to re-type")


def add_supplies(graph: CircuitGraph, supplies: dict) -> None:
    """Drive a named rail the drawing labels but never gives a source.

    A drafter writes ``VCC`` at the top of a rail and leaves the supply
    implied -- it lives on the next sheet, or in their head. The netlist
    cannot: an undriven rail floats, and LTspice solves it as whatever the
    op-amp's input leakage decides. Nothing in the drawing says what the
    voltage is, so this is never inferred; it is only ever what the caller
    asked for, and it is recorded as a warning so the output cannot be
    mistaken for something that was read off the page.
    """
    if not supplies:
        return
    by_name = {v.upper(): k for k, v in graph.net_names.items()}
    for key, value in supplies.items():
        key = str(key).strip()
        node = by_name.get(key.upper())
        if node is None and key.lstrip("-").isdigit():
            node = int(key)
        if node is None:
            graph.warnings.append(f"no net called {key!r} to supply")
            continue
        if node == graph.ground_node:
            graph.warnings.append("refusing to put a supply across ground")
            continue
        taken = {c.ref.upper() for c in graph.components}
        i = 1
        while f"V{i}" in taken:
            i += 1
        # Placed clear of the drawing; connectivity is carried by net labels,
        # so where it sits on the sheet is cosmetic.
        xs = [c.det.cx for c in graph.components] or [0.0]
        ys = [c.det.cy for c in graph.components] or [0.0]
        w = max(20.0, 0.04 * (max(xs) - min(xs) + 1))
        x, y = min(xs) - 2.5 * w, max(ys) + 2.0 * w
        det = Detection("source_voltage", x - w / 2, y - w / 2,
                        x + w / 2, y + w / 2)
        graph.components.append(Component(
            ref=f"V{i}", cls="source_voltage", det=det,
            nodes=[node, graph.ground_node], value=str(value),
            orientation="R0"))
        graph.warnings.append(
            f"added V{i} = {value} from {graph.node_name(node)} to ground "
            f"by hand -- the drawing does not say this")


def set_pins(graph: CircuitGraph, mapping: dict) -> None:
    """State a part's node order outright, by net name or number.

    The last resort, and the only lever that can express an arbitrary
    permutation: ``--flip`` reverses, which on a three-pin part swaps the
    outer two and can never move the middle one. A FET whose drain and gate
    came out transposed needs exactly that, and there is no reader for a FET's
    pins -- see :func:`build_circuit`.

    Nets are named as the netlist names them (``VCC``, ``N005``, ``0``), so
    what is typed is what is read off the printed netlist.
    """
    if not mapping:
        return
    by_name = {v.upper(): k for k, v in graph.net_names.items()}
    want = {k.upper(): v for k, v in mapping.items()}
    for comp in graph.components:
        spec_str = want.pop(comp.ref.upper(), None)
        if spec_str is None:
            continue
        names = [t.strip() for t in str(spec_str).split(",") if t.strip()]
        sp = spec(comp.cls)
        # A switch may be given four: its two terminals, then the control
        # pair LTspice's ``sw`` has and the drawing never shows. Without this
        # the control pins can only ever be ground, which is to say the switch
        # can only ever be open -- a circuit that simulates and does nothing.
        n_ctrl = 2 if (comp.cls == "switch" and len(names) == sp.n_ports + 2) else 0
        if len(names) != sp.n_ports + n_ctrl:
            graph.warnings.append(
                f"{comp.ref} takes {sp.n_ports} nodes"
                + (" (or 4, with the control pair)" if comp.cls == "switch" else "")
                + f", got {len(names)}")
            continue
        nodes, bad = [], None
        for nm in names:
            if nm == "0" or nm.upper() == "GND":
                nodes.append(graph.ground_node)
            elif nm.upper() in by_name:
                nodes.append(by_name[nm.upper()])
            elif nm.upper().startswith("N") and nm[1:].isdigit():
                nodes.append(int(nm[1:]))
            elif nm.lstrip("-").isdigit():
                nodes.append(int(nm))
            else:
                bad = nm
                break
        if bad is not None:
            graph.warnings.append(f"no net called {bad!r} for {comp.ref}")
            continue
        if n_ctrl:
            comp.control = nodes[sp.n_ports:]
            nodes = nodes[:sp.n_ports]
        comp.nodes = nodes
        graph.warnings.append(
            f"set {comp.ref} pins to "
            f"{', '.join(graph.node_name(n) for n in graph.wired_nodes(comp))}"
            f" by hand")
    for ref in want:
        graph.warnings.append(f"no component called {ref} to set pins on")


#: A ground symbol covered this much by a more confident terminal is taken to
#: be that terminal, seen twice. Measured: the pair that forced this overlapped
#: at 0.749 against the detector's own cross-class NMS threshold of 0.75 --
#: it survived by a thousandth.
SHADOWED_GND_IOU = 0.50


def _iou(a, b) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _shadowed_grounds(detections: list) -> set:
    """``id()`` of ground symbols that are really a terminal, seen twice.

    **A ground symbol is the only detection whose error is not local.** Every
    one of them is unioned onto a single node, so one false ground does not
    add a wrong part -- it shorts together any two nets that happen to carry
    one, anywhere on the page. A wrong resistor is a wrong resistor; a wrong
    ground is a different circuit.

    That asymmetry is why this rule is asymmetric. A terminal and a ground are
    the same kind of thing to look at, a small symbol on the end of a wire,
    and the detector duly returns both for one drawing of either. Measured on
    a VCC rail: ``terminal`` at 0.73 and ``gnd`` at 0.26 over the same ink,
    overlapping at 0.749 where cross-class NMS cuts at 0.75. The rail was
    welded to ground, the supply shorted, and the only visible symptom was a
    source with both terminals on node 0.

    Raising the NMS threshold would fix that pair and tune the detector on one
    example. Refusing to let a barely-believed ground outvote a confident
    terminal fixes the class of problem, and costs nothing anywhere else.
    """
    grounds = [d for d in detections if d.cls == "gnd"]
    terminals = [d for d in detections if d.cls == "terminal"]
    out = set()
    for g in grounds:
        for t in terminals:
            if t.conf > g.conf and _iou(g, t) >= SHADOWED_GND_IOU:
                out.add(id(g))
                break
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
    left and right of its body is R90. R0 against R180 is left undecided: the
    schematic is wired from the node order, which for a diode ``diode_net``
    now sets, so the result is electrically right either way and only the
    direction the triangle happens to point is cosmetic.
    """
    if len(terms) < 2:
        return "R0" if det.h >= det.w else "R90"
    # First and last, not first and second: after pin ordering those are the
    # collector and emitter, which are the device's axis. For a two-terminal
    # part they are the only two anyway.
    (_, x_a, y_a), (_, x_b, y_b) = terms[0], terms[-1]
    return "R90" if abs(x_a - x_b) > abs(y_a - y_b) else "R0"


def build_circuit(
    bgr: np.ndarray,
    detections: list,
    *,
    node_map: NodeMap | None = None,
    values: dict | None = None,
    bridge_px: int | None = 6,
    method: str = "flatfield",
    drop_outside: str = "none",
    thin_q: float | None = 50.0,
    reconnect_px: float = 24.0,
    restore_branches: bool = True,
    fuse_stranded_ground: bool = True,
    unet_thr: float | None = None,
) -> tuple[CircuitGraph, NodeMap]:
    """Turn detections into a :class:`CircuitGraph`."""
    if node_map is None:
        node_map = build_node_map(bgr, detections, bridge_px=bridge_px, method=method,
                                  drop_outside=drop_outside, thin_q=thin_q,
                                  reconnect_px=reconnect_px,
                                  restore_branches=restore_branches,
                                  unet_thr=unet_thr)

    warnings: list[str] = []
    uf = UnionFind()
    for n in range(1, node_map.n_nodes + 1):
        uf.find(n)

    # Every ground symbol names the same electrical node, so tie them together.
    #
    # A ground symbol is a connection point, not a component that separates
    # nets: everything touching it belongs to one node. That matters because
    # the symbol is erased before tracing, and erasing it severs any wire
    # running through it, stranding the ground stub from the rail it belongs
    # to. Taking only the strongest fragment leaves that break in place, so we
    # union every fragment the symbol touches, exactly as a junction does.
    ground_labels: list[int] = []
    terminal_hits: dict = {}        # id(terminal detection) -> a node label
    shadowed = _shadowed_grounds(detections)
    if shadowed:
        warnings.append(
            f"ignored {len(shadowed)} ground symbol(s) sitting on a more "
            f"confident terminal -- a false ground shorts every net it touches"
        )
    for d in detections:
        if d.cls not in ("gnd", "terminal") or id(d) in shadowed:
            continue
        k = max(4, int(0.35 * max(d.w, d.h)))
        touching = [
            lab for lab, st in _ring_label_stats(node_map.labels, d.box, k).items()
            if st[0] >= 3
        ]
        if not touching:
            continue
        for other in touching[1:]:
            uf.union(touching[0], other)
        if d.cls == "gnd":
            ground_labels.append(touching[0])
        else:
            terminal_hits[id(d)] = touching[0]
    for other in ground_labels[1:]:
        uf.union(ground_labels[0], other)

    # Resolve each placeable component to its terminals.
    raw: list[tuple] = []
    counters: dict = {}
    diode_states: list[str] = []
    opamp_states: list[str] = []
    fet_states: list[str] = []
    source_states: list[str] = []
    for d in sorted(detections, key=lambda d: (d.cy, d.cx)):
        if d.cls not in PLACEABLE:
            continue
        sp = spec(d.cls)
        terms = _terminals(node_map, d, sp.n_ports)
        if d.cls == "transistor_bjt":
            # Order is C B E for SPICE, and the drawing says which is which.
            terms = order_transistor_terminals(node_map.binary, d, terms)
        elif d.cls == "diode":
            # Order is anode, cathode. No ink statistic separates the bar from
            # the triangle's base (four were measured, the best reached 76%),
            # so this one is read by a small learned model -- and like the
            # transistor above, it declines rather than guesses, leaving the
            # order alone when the drawing does not say. The grey image is
            # passed, not the binary mask: the model is trained on the
            # photograph, where the stroke's weight survives.
            nfo: dict = {}
            terms = order_diode_terminals(bgr, d, terms, info=nfo)
            diode_states.append(nfo.get("state", "no_reading"))
        elif d.cls == "opamp":
            # Order is invin, noninvin, out -- LTspice's own SpiceOrder, not
            # the "In+ In- Out" that is widely repeated. Until this ran the
            # three nets went out in order of skeleton pixel count, which is
            # no order at all, and an op-amp wired through the wrong pins
            # still opens, still verifies and still simulates.
            nfo = {}
            terms = order_opamp_terminals(node_map.binary, node_map.labels,
                                          d, terms, info=nfo)
            opamp_states.append(nfo.get("state", "no_reading"))
        elif d.cls in ("source_voltage", "source_current"):
            # The + and - are drawn inside the circle; nobody was reading
            # them, so half the sources in any drawing came out backwards.
            nfo = {}
            terms = order_source_terminals(node_map.binary, d, terms, info=nfo)
            source_states.append(nfo.get("state", "no_reading"))
        elif d.cls == "transistor_fet":
            # The gate is read; drain and source are a convention the symbol
            # does not fix. See ``order_fet_terminals``.
            nfo = {}
            terms = order_fet_terminals(d, terms, info=nfo)
            fet_states.append(nfo.get("state", "no_reading"))
        if len(terms) < sp.n_ports:
            warnings.append(
                f"{d.cls} at ({d.cx:.0f},{d.cy:.0f}) found only "
                f"{len(terms)}/{sp.n_ports} terminals; left unconnected"
            )
        counters[sp.spice_prefix] = counters.get(sp.spice_prefix, 0) + 1
        ref = f"{sp.spice_prefix}{counters[sp.spice_prefix]}"
        raw.append((ref, d, sp, terms))

    # Say what the diode stage did. A stage that quietly declines looks exactly
    # like one that is not installed, and the netlist is wrong either way --
    # the diode is simply the wrong way round, which simulates without
    # complaint. This is the same lesson as the Ollama reader returning empty
    # strings and as LTspice opening a schematic whose wires do not meet.
    if diode_states:
        read = sum(s == "read" for s in diode_states)
        if all(s == "no_model" for s in diode_states):
            warnings.append(
                f"diode polarity not read ({len(diode_states)} diode(s)): no "
                "model installed. Build one with "
                "`python scripts/diode_orient.py --build --train`, or set "
                "CIRCUIT2SPICE_DIODE. Terminal order is positional until then"
            )
        elif read == len(diode_states):
            warnings.append(
                f"read the polarity of {read} diode(s) from the drawing")
        else:
            warnings.append(
                f"read the polarity of {read}/{len(diode_states)} diode(s) "
                f"from the drawing; the rest kept positional order "
                f"({', '.join(sorted(set(s for s in diode_states if s != 'read')))})"
            )

    # And say what the op-amp stage did, for the same reason. This one matters
    # more than it looks: the inputs decide whether the feedback is negative or
    # positive, so a swap turns an amplifier into a latch, and every headline
    # metric in this project -- isomorphism, BoM, even the LTspice round-trip
    # -- stays green while it happens.
    if opamp_states:
        read = sum(s.startswith("read") for s in opamp_states)
        shared = sum("shared" in s for s in opamp_states)
        if read == len(opamp_states):
            warnings.append(
                f"read the pin roles of {read} op-amp(s) from the drawing"
                + (f" ({shared} with a pin shared, e.g. a unity-gain buffer)"
                   if shared else "")
            )
        else:
            warnings.append(
                f"read the pin roles of {read}/{len(opamp_states)} op-amp(s) "
                f"from the drawing; the rest kept positional order "
                f"({', '.join(sorted(set(s for s in opamp_states if not s.startswith('read'))))})"
            )

    if source_states:
        ok = sum(x == "read" for x in source_states)
        if ok:
            warnings.append(
                f"read the polarity of {ok}/{len(source_states)} source(s) "
                f"from the drawing")
        if ok < len(source_states):
            warnings.append(
                f"{len(source_states) - ok} source(s) kept positional "
                f"terminal order -- reverse one with --flip if it is backwards")

    if fet_states:
        ok = sum(s == "gate_read" for s in fet_states)
        if ok:
            warnings.append(
                f"read the gate of {ok} FET(s) from the drawing; drain and "
                f"source follow the usual convention, which the symbol itself "
                f"cannot confirm -- reverse one with --flip if it is backwards"
            )
        if ok < len(fet_states):
            warnings.append(
                f"{len(fet_states) - ok} FET(s) kept positional "
                f"drain/gate/source order (no clean gate lead)"
            )

    # Decide which node is ground, then renumber so ground is 0.
    degree: dict = {}
    for _ref, _d, _sp, terms in raw:
        for lab, _cx, _cy in terms:
            r = uf.find(lab)
            degree[r] = degree.get(r, 0) + 1

    if ground_labels:
        ground_root = uf.find(ground_labels[0])
        # A ground symbol whose node carries only one terminal is impossible:
        # nothing can flow through it. What it means is that the drafter used
        # the symbol as an *implicit* tie to a return rail they drew without
        # one -- the rail carries the source's other terminal and no ground
        # flag of its own, and the reference schematic fuses the two. Nothing
        # in the drawing connects them, so wire tracing cannot find it; the
        # only evidence is the stranded stub itself.
        #
        # Rejoin it with the node that the ground-less fallback would have
        # picked anyway -- the highest-degree node -- breaking ties on wire
        # length, because a return rail is the longest conductor on the page.
        if fuse_stranded_ground and degree.get(ground_root, 0) < 2:
            others = [r for r in degree if r != ground_root]
            if others:
                counts = np.bincount(
                    node_map.labels.ravel(), minlength=node_map.n_nodes + 1
                )
                px: dict = {}
                for lab in range(1, node_map.n_nodes + 1):
                    r = uf.find(lab)
                    px[r] = px.get(r, 0) + int(counts[lab])
                best = max(others, key=lambda r: (degree[r], px.get(r, 0)))
                uf.union(ground_root, best)
                ground_root = uf.find(ground_root)
                warnings.append(
                    "ground symbol was left on a node of its own; fused it with "
                    "the highest-degree node, which is the return rail"
                )
    else:
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
        terminal_nodes={k: node_id(lab) for k, lab in terminal_hits.items()},
    )
    return graph, node_map
