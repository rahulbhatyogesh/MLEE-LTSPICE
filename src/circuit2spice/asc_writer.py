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
    by_node: dict = {}
    for comp in graph.components:
        if comp.ref not in placement:
            continue
        _origin, _rot, pins = placement[comp.ref]
        cx = sum(p[0] for p in pins) / len(pins)
        cy = sum(p[1] for p in pins) / len(pins)
        for node, pin in zip(comp.nodes, pins):
            if node >= 0:
                by_node.setdefault(node, []).append((pin, _escape(pin, (cx, cy))))

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
