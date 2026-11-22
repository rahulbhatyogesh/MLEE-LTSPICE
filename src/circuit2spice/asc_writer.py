"""Write an LTspice ``.asc`` schematic from a :class:`CircuitGraph`.

Three facts about the ``.asc`` format drive this module, each verified against
LTspice 26.1.1 by netlisting probe schematics with ``LTspice.exe -netlist``:

1. **Symbol origins must sit on the 16-unit grid.** An off-grid symbol is
   silently snapped, its pins move, and wires that look attached are not.
2. **Pin offsets come from the symbol file, not the drawn body.** A resistor's
   second pin is at ``(16, 96)``, past the end of the drawn zig-zag.
3. **Crossing wires do not connect.** Only shared endpoints, and endpoints
   landing on another wire, create a net. This is what makes simple
   hub-and-spoke routing safe: routes may cross freely.

Ground needs no routing at all -- a ``FLAG x y 0`` placed directly on a pin ties
it to node 0.

The resulting layout is electrically faithful but not beautiful: components keep
their relative positions from the photograph and nets are routed as L-shaped
spokes to a per-net hub. Producing a draughtsman-quality schematic is a
placement-and-routing problem in its own right and is out of scope.
"""

from __future__ import annotations

from pathlib import Path

from .classes import MODEL_CARDS, SYMBOL_LIBS, spec
from .graph import CircuitGraph
from .netlist import DEFAULT_ANALYSIS
from .symbols import load_symbol

GRID = 16
#: Target on-sheet spacing between neighbouring components.
TARGET_PITCH = 224


def snap(v: float, grid: int = GRID) -> int:
    return int(round(v / grid)) * grid


def _layout_scale(graph: CircuitGraph) -> float:
    """Pick an image->sheet scale that spreads components about TARGET_PITCH apart."""
    pts = [(c.det.cx, c.det.cy) for c in graph.components]
    if len(pts) < 2:
        return 1.0
    best = []
    for i, (xi, yi) in enumerate(pts):
        d = min(
            ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            for j, (xj, yj) in enumerate(pts)
            if j != i
        )
        best.append(d)
    best.sort()
    median_nn = best[len(best) // 2] or 1.0
    return max(0.25, min(8.0, TARGET_PITCH / median_nn))


def build_placement(graph: CircuitGraph) -> dict:
    """ref -> (origin_xy, orientation, [pin_xy, ...]) in sheet coordinates."""
    scale = _layout_scale(graph)
    xs = [c.det.cx for c in graph.components] or [0.0]
    ys = [c.det.cy for c in graph.components] or [0.0]
    min_x, min_y = min(xs), min(ys)

    placement: dict = {}
    for comp in graph.components:
        sym_name = spec(comp.cls).ltspice_symbol
        if not sym_name:
            continue
        sym = load_symbol(sym_name)
        offs = sym.pin_offsets(comp.orientation)
        # Centre the symbol on the component's position by subtracting the
        # centroid of its (rotated) pins.
        mx = sum(o[0] for o in offs) / len(offs)
        my = sum(o[1] for o in offs) / len(offs)
        cx = (comp.det.cx - min_x) * scale + 160
        cy = (comp.det.cy - min_y) * scale + 128
        origin = (snap(cx - mx), snap(cy - my))
        pins = [(origin[0] + dx, origin[1] + dy) for dx, dy in offs]
        placement[comp.ref] = (origin, comp.orientation, pins)
    return placement


ESCAPE = 32


def _escape(pin: tuple, centroid: tuple) -> tuple:
    """A point just outside the symbol, on the pin's own outward axis.

    Routing straight from a pin towards the net hub is unsafe: on a rotated
    (horizontal) component both pins share a y, so a horizontal first segment
    runs through the *other* pin and shorts the two nets together. Stepping
    outward along the pin axis first, then turning, keeps each spoke clear of
    its sibling pins.
    """
    px, py = pin
    dx, dy = px - centroid[0], py - centroid[1]
    if dx == 0 and dy == 0:
        return px, py - ESCAPE
    if abs(dx) >= abs(dy):
        return px + (ESCAPE if dx > 0 else -ESCAPE), py
    return px, py + (ESCAPE if dy > 0 else -ESCAPE)


def _escape_candidates(pin: tuple, centroid: tuple):
    """Escape points to try, best-looking first.

    The preferred stub runs straight out along the pin's own axis. The
    alternatives turn 90 degrees instead, which is always clear of the sibling
    pins -- on a vertical symbol the siblings differ in *y*, so a horizontal
    stub cannot cross one, and vice versa -- and then lengthen the stub.
    """
    px, py = pin
    dx, dy = px - centroid[0], py - centroid[1]
    if dx == 0 and dy == 0:
        dy = -1.0
    if abs(dx) >= abs(dy):
        out = (1 if dx > 0 else -1, 0)
    else:
        out = (0, 1 if dy > 0 else -1)
    perp = (out[1], out[0])
    for dist in (ESCAPE, 2 * ESCAPE, 3 * ESCAPE):
        yield px + out[0] * dist, py + out[1] * dist
        yield px + perp[0] * dist, py + perp[1] * dist
        yield px - perp[0] * dist, py - perp[1] * dist
    yield px + out[0] * GRID, py + out[1] * GRID


def _segment_points(a: tuple, b: tuple) -> list:
    """Every grid point on an axis-aligned segment, endpoints included.

    Everything the writer emits is snapped to :data:`GRID`, so walking the
    segment in grid steps enumerates exactly the points at which LTspice can
    make a connection.
    """
    (ax, ay), (bx, by) = a, b
    if ax == bx and ay == by:
        return [a]
    step = GRID if ax == bx else GRID
    if ax == bx:
        lo, hi = sorted((ay, by))
        return [(ax, y) for y in range(lo, hi + 1, step)] + [(ax, hi)]
    lo, hi = sorted((ax, bx))
    return [(x, ay) for x in range(lo, hi + 1, step)] + [(hi, ay)]


def _assign_escapes(graph: CircuitGraph, placement: dict) -> tuple[dict, list]:
    """Pick a stub endpoint per pin such that no two nets ever touch.

    Connectivity here is carried by net-name ``FLAG``s on short stubs, which is
    exact only as long as two *different* nets never share a point. Nothing in
    the geometry guarantees that: two symbols stacked a pitch apart in the same
    column have their facing pins escape towards each other and meet in the
    middle, which both stamps two names on one point and fuses the two stubs
    into a single wire. LTspice then silently reports one of the nets wired to
    the other -- the exact failure the round-trip check exists to catch.

    So stubs are claimed point by point: a candidate is accepted only if every
    grid point it occupies is free or already belongs to the same net.

    Returns ``{(ref, pin_index): escape_point}`` and a list of warnings for any
    pin where no clear route existed.
    """
    claimed: dict = {}
    warnings: list = []

    # Pins are claimed first: they are fixed, so a stub must route around them.
    pin_xy: list = []
    for comp in graph.components:
        if comp.ref not in placement:
            continue
        _origin, _rot, pins = placement[comp.ref]
        cx = sum(p[0] for p in pins) / len(pins)
        cy = sum(p[1] for p in pins) / len(pins)
        for i, (node, pin) in enumerate(zip(graph.wired_nodes(comp), pins)):
            if node < 0:
                continue
            claimed.setdefault(pin, node)
            pin_xy.append((comp.ref, i, node, pin, (cx, cy)))

    escapes: dict = {}
    # Longest stubs are hardest to place, but all stubs start equal; order by
    # position so the output is deterministic.
    for ref, i, node, pin, centroid in sorted(pin_xy, key=lambda t: (t[3], t[0], t[1])):
        chosen = None
        for cand in _escape_candidates(pin, centroid):
            pts = _segment_points(pin, cand)
            if all(claimed.get(p, node) == node for p in pts):
                chosen = cand
                for p in pts:
                    claimed[p] = node
                break
        if chosen is None:
            chosen = _escape(pin, centroid)
            warnings.append(
                f"{ref} pin {i}: no clear stub route; the .asc may short "
                f"{graph.node_name(node)} to a neighbouring net"
            )
        escapes[(ref, i)] = chosen
    return escapes, warnings


def _route(spokes: list, hub: tuple) -> list:
    """Escape stub, then a perpendicular-first L to the shared hub."""
    hx, hy = hub
    wires = []
    for (px, py), (ex, ey) in spokes:
        if (px, py) != (ex, ey):
            wires.append((px, py, ex, ey))
        # Vertical first, so the horizontal leg sits on the net's own row
        # rather than along the component body.
        if ey != hy:
            wires.append((ex, ey, ex, hy))
        if ex != hx:
            wires.append((ex, hy, hx, hy))
    return wires


def to_asc(
    graph: CircuitGraph,
    *,
    analysis: str = DEFAULT_ANALYSIS,
    placement: dict | None = None,
) -> str:
    """Render the circuit as LTspice ``.asc`` text."""
    placement = placement or build_placement(graph)

    # Gather every pin, with its escape point, grouped by electrical node.
    escapes, esc_warnings = _assign_escapes(graph, placement)
    by_node: dict = {}
    for comp in graph.components:
        if comp.ref not in placement:
            continue
        _origin, _rot, pins = placement[comp.ref]
        for i, (node, pin) in enumerate(zip(graph.wired_nodes(comp), pins)):
            if node >= 0:
                by_node.setdefault(node, []).append((pin, escapes[(comp.ref, i)]))
    for w in esc_warnings:
        if w not in graph.warnings:
            graph.warnings.append(w)

    # Connectivity is expressed with net *labels*, not routed wires.
    #
    # Geometric routing is where this goes wrong: an L-shaped spoke to a net
    # hub puts a wire endpoint at a corner, and if that corner happens to land
    # on another net's segment LTspice silently fuses the two nets.  Staggering
    # channels only reduces the chance.  A FLAG carrying the net name is exact
    # -- it is the same mechanism ground already uses -- so every pin gets a
    # short outward stub ending in its net label.  The schematic reads like a
    # labelled EDA drawing and the connectivity is correct by construction.
    wires: list = []
    flags: list = []
    for node, spokes in sorted(by_node.items()):
        name = graph.node_name(node)
        for (px, py), (ex, ey) in spokes:
            if (px, py) != (ex, ey):
                wires.append((px, py, ex, ey))
            flags.append((ex, ey, name))

    all_x = [p[0] for sp in by_node.values() for p, _e in sp] or [0]
    all_y = [p[1] for sp in by_node.values() for p, _e in sp] or [0]
    sheet_w = max(880, snap(max(all_x) + 320))
    sheet_h = max(680, snap(max(all_y) + 320))

    out: list = ["Version 4", f"SHEET 1 {sheet_w} {sheet_h}"]
    for x1, y1, x2, y2 in wires:
        out.append(f"WIRE {x1} {y1} {x2} {y2}")
    for x, y, name in flags:
        out.append(f"FLAG {x} {y} {name}")

    for comp in graph.components:
        if comp.ref not in placement:
            continue
        (ox, oy), rot, _pins = placement[comp.ref]
        sym_name = spec(comp.cls).ltspice_symbol
        out.append(f"SYMBOL {sym_name} {ox} {oy} {rot}")
        out.append(f"SYMATTR InstName {comp.ref}")
        if comp.value:
            out.append(f"SYMATTR Value {comp.value}")

    # SPICE directives live as "!"-prefixed TEXT entries.
    ty = sheet_h - 160
    seen: set = set()
    for comp in graph.components:
        if comp.value in MODEL_CARDS and comp.value not in seen:
            out.append(f"TEXT 64 {ty} Left 2 !{MODEL_CARDS[comp.value]}")
            seen.add(comp.value)
            ty += 32
        lib = SYMBOL_LIBS.get(spec(comp.cls).ltspice_symbol or "")
        if lib and lib not in seen:
            out.append(f"TEXT 64 {ty} Left 2 !.lib {lib}")
            seen.add(lib)
            ty += 32
    out.append(f"TEXT 64 {ty} Left 2 !{analysis}")

    # LTspice writes CRLF; match it so the file is byte-normal for the tool.
    return "\r\n".join(out) + "\r\n"


def write_asc(graph: CircuitGraph, path: str | Path, **kw) -> Path:
    path = Path(path)
    path.write_text(to_asc(graph, **kw), encoding="ascii", newline="")
    return path
