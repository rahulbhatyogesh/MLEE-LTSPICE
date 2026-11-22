"""Which lead is the emitter? Read the bar and the arrowhead.

``_terminals()`` finds *where* a component's wires leave its box, ranked by how
much skeleton each contact carries. For a two-terminal part that is enough --
a resistor is symmetric. For a transistor it is not: SPICE writes ``Q<ref> C B
E``, and getting that order wrong yields a netlist that is topologically
identical to the right one but electrically nothing like it. An NPN with its
collector and emitter transposed sits in reverse-active mode, where beta is a
handful rather than a hundred, so the bias point is wrong and the one thing the
pipeline promises -- a schematic that simulates -- quietly fails.

Nothing upstream notices, because graph isomorphism over an undirected
component/net graph cannot see pin roles. The drawing can, in two steps.

**The base is the lead the bar lies across.** Every BJT symbol runs its base
into the middle of the flat bar, perpendicular to it, while the collector and
emitter attach at the bar's two ends. So a probe swept across the base lead
meets a long stroke and the other two do not. The probe has to *slide* along
the lead: the bar is not at the symbol's centre, and on some drawings it sits
hard against the edge of the box.

Angles cannot do this job. The three leads commonly leave the box about 120
degrees apart, and in that configuration every direction-based rule --
"most opposite the other two", "most perpendicular to their axis" -- scores the
collector as well as it scores the base, so it is a coin toss.

**The emitter is the lead the arrowhead is on.** An arrowhead is the thickest
ink in the symbol, so the peak of the distance transform locates it. But its
*bearing* from the centre is not the lead's bearing -- the arrow sits near the
bar while the terminal is out at the box edge, and the lead kinks in between,
which on CGHD's drawings puts the two up to 70 degrees apart. What does survive
is which side of the bar it is on: project the arrow and the two candidate
terminals onto the bar's axis, and the emitter is the one sharing the arrow's
sign.
"""

from __future__ import annotations

import cv2
import numpy as np

#: How far along each lead to sweep the crossing probe, as a fraction of the
#: distance from the symbol centre to the terminal.
SWEEP = (0.10, 0.75)
SWEEP_STEPS = 9

#: Half-length of the crossing probe, as a fraction of the symbol's size.
PROBE_FRAC = 0.30

#: The bar must out-cross the other leads by this factor before it is
#: believed. Measured margins on CGHD run 1.5x-4x, so this is loose.
MIN_BAR_RATIO = 1.15

#: The thickest ink must exceed the median stroke by this factor to count as
#: an arrowhead. Calibrated, not guessed: on CGHD's transistors the peak runs
#: 4.8x-6.3x the median, while two plain strokes merely crossing reach 2.7x,
#: so anything in between separates them.
MIN_ARROW_RATIO = 3.0


def _density(binary: np.ndarray, p0, p1, half_width: float) -> float:
    """Ink fraction in a thin rectangle laid along ``p0``–``p1``."""
    h, w = binary.shape[:2]
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy))
    if length < 4.0:
        return 0.0
    n_along = max(8, int(length))
    n_across = max(3, int(half_width * 2) + 1)
    t = np.linspace(0.0, 1.0, n_along)[:, None]
    s = np.linspace(-half_width, half_width, n_across)[None, :]
    px, py = -dy / length, dx / length
    xs = np.rint(x0 + t * dx + s * px).astype(int)
    ys = np.rint(y0 + t * dy + s * py).astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not ok.any():
        return 0.0
    return float((binary[ys[ok], xs[ok]] > 0).mean())


def _crossing(binary: np.ndarray, det, term) -> float:
    """Strongest stroke lying *across* this lead, anywhere along it."""
    dx, dy = term[1] - det.cx, term[2] - det.cy
    n = float(np.hypot(dx, dy))
    if n < 1e-6:
        return 0.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    r = PROBE_FRAC * max(det.w, det.h)
    best = 0.0
    for f in np.linspace(*SWEEP, SWEEP_STEPS):
        cx, cy = det.cx + ux * n * f, det.cy + uy * n * f
        best = max(best, _density(binary, (cx - px * r, cy - py * r),
                                  (cx + px * r, cy + py * r), 2.0))
    return best


def _arrow_point(binary: np.ndarray, det):
    """Centre of the thickest ink inside the symbol, or ``None``."""
    h, w = binary.shape[:2]
    x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
    x2, y2 = min(w, int(det.x2)), min(h, int(det.y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = (binary[y1:y2, x1:x2] > 0).astype(np.uint8)
    if not crop.any():
        return None
    dt = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
    ink = dt[crop > 0]
    peak = float(dt.max())
    typical = float(np.median(ink)) if ink.size else 0.0
    if typical <= 0 or peak < MIN_ARROW_RATIO * typical:
        return None                      # a plain line, no arrowhead to read
    iy, ix = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return float(x1 + ix), float(y1 + iy)


def order_transistor_terminals(binary: np.ndarray, det, terms: list) -> list:
    """Reorder ``terms`` into SPICE's ``(collector, base, emitter)``.

    Returns the input unchanged whenever the drawing does not actually say --
    not three terminals, no bar that stands out, no arrowhead thicker than the
    strokes around it, or an arrow that lands on the bar's axis rather than to
    one side of it. A known-arbitrary order is better than a confidently wrong
    one, and the caller cannot tell the two apart afterwards.
    """
    if binary is None or len(terms) != 3:
        return terms

    cross = [_crossing(binary, det, t) for t in terms]
    b = int(np.argmax(cross))
    rest = [i for i in range(3) if i != b]
    if cross[b] < MIN_BAR_RATIO * max(cross[i] for i in rest):
        return terms                     # no lead clearly carries the bar

    arrow = _arrow_point(binary, det)
    if arrow is None:
        return terms

    # The bar runs across the base lead, so its axis is that lead's normal.
    bx, by = terms[b][1] - det.cx, terms[b][2] - det.cy
    n = float(np.hypot(bx, by))
    if n < 1e-6:
        return terms
    ax, ay = -by / n, bx / n

    along = {i: (terms[i][1] - det.cx) * ax + (terms[i][2] - det.cy) * ay
             for i in rest}
    s_arrow = (arrow[0] - det.cx) * ax + (arrow[1] - det.cy) * ay
    lo, hi = sorted(rest, key=lambda i: along[i])
    if along[lo] >= 0 or along[hi] <= 0:
        return terms                     # both leads the same side: no call
    emitter = hi if s_arrow > 0 else lo
    collector = lo if emitter == hi else hi
    return [terms[collector], terms[b], terms[emitter]]


# --------------------------------------------------------------------------
# One lead alone on one side: gate, base, or an op-amp's output
# --------------------------------------------------------------------------

#: The lone lead must clear the other two by this fraction of the symbol's
#: size along the axis before the split is believed.
LONE_SPLIT_FRAC = 0.25

#: ...and must sit between them across it, or three leads spread 120 degrees
#: apart read as a 2-vs-1 split on whichever axis happens to suit.
LONE_OFFSET_FRAC = 0.35


def lone_lead_direction(det, terms: list, min_split_frac: float = LONE_SPLIT_FRAC,
                        max_offset_frac: float = LONE_OFFSET_FRAC):
    """Unit vector pointing from the pair of leads towards the lone one.

    Three-terminal symbols in this dataset are all built the same way: one
    lead leaves on its own and two leave together from the opposite side.
    CGHD's ``classes_ports.json`` states it outright --

        transistor.fet   gate [0, 0.5]   drain [0.5, 0]   source [0.5, 1]
        transistor.bjt   base [0, 0.5]   collector [0.7, 1]  emitter [0.7, 0]

    -- and an op-amp's output leaves the apex while both inputs enter the
    base. So the lone lead is the gate, the base, or the output, and finding
    it needs no ink at all.

    Returns ``None`` unless the split is clean: the lone lead must stand off
    from the pair by more than they are spread apart, and must sit *between*
    them across the axis. Without that second test three leads at 120 degrees
    -- the usual way a transistor is drawn -- read as a 2-vs-1 split on
    whichever axis happens to suit, which is a confident answer to a question
    the drawing never answered.
    """
    if len(terms) != 3:
        return None
    best = None
    for a in (0, 1):
        size = det.w if a == 0 else det.h
        if size <= 0:
            continue
        across_size = det.h if a == 0 else det.w
        proj = [(t[1] - det.cx) if a == 0 else (t[2] - det.cy) for t in terms]
        cross = [(t[2] - det.cy) if a == 0 else (t[1] - det.cx) for t in terms]
        order = sorted(range(3), key=lambda i: proj[i])
        for lone, pair, s in ((order[0], order[1:], -1.0),
                              (order[2], order[:2], +1.0)):
            gap = min(abs(proj[lone] - proj[p]) for p in pair)
            spread = abs(proj[pair[0]] - proj[pair[1]])
            if gap < min_split_frac * size or gap <= spread:
                continue
            mid = 0.5 * (cross[pair[0]] + cross[pair[1]])
            if abs(cross[lone] - mid) > max_offset_frac * max(across_size, 1):
                continue
            score = gap / size
            if best is None or score > best[0]:
                best = (score, a, s)
    if best is None:
        return None
    _score, a, s = best
    return (s, 0.0) if a == 0 else (0.0, s)


def order_fet_terminals(det, terms: list, info=None) -> list:
    """Reorder ``terms`` into SPICE's ``(drain, gate, source)``.

    **The gate is read; the drain and source are a convention.** That split is
    the whole point of this function, and the symbol is what forces it.

    The gate is the lone lead, by :func:`lone_lead_direction`, and that is a
    real reading: it is wrong only if the drawing puts two leads on the gate
    side, which no FET symbol does. Getting it right matters more than it
    looks, because the gate is SPICE's *middle* node -- so once it is in the
    middle, drain and source are the two ends and ``--flip M1`` swaps exactly
    those two. Before this they were ordered by skeleton pixel count, where
    the gate could land anywhere and no single flag could fix it.

    Drain against source is not in the ink. The common 3-terminal MOSFET
    symbol is a gate plate, a channel bar and two leads off the channel, and
    it is *symmetric* about the channel: no arrowhead, no bulk connection,
    nothing that distinguishes one end from the other. (A BJT has the emitter
    arrow, which is why :func:`order_transistor_terminals` can read that one.)
    So the order here comes from the convention CGHD's own port table records
    -- ``drain [0.5, 0]`` above ``source [0.5, 1]``, drain on the side reached
    by turning the gate direction a quarter turn -- and a drawing that mirrors
    its FET will come out reversed. It is stated as a convention in the
    warnings, not reported as a reading, because the two are not the same
    claim and the caller cannot tell them apart afterwards.
    """
    def note(state, **kw):
        if info is not None:
            info.update(state=state, **kw)

    if len(terms) != 3:
        note("not_three_terminals")
        return terms
    u = lone_lead_direction(det, terms)          # body -> gate
    if u is None:
        note("no_gate")
        return terms

    along = [(t[1] - det.cx) * u[0] + (t[2] - det.cy) * u[1] for t in terms]
    g = int(max(range(3), key=lambda i: along[i]))
    rest = [i for i in range(3) if i != g]

    # Drain is a quarter turn from the gate direction: with the gate to the
    # left, (-1, 0), that is (0, -1), which is up -- the way a FET is drawn
    # when nobody has mirrored it.
    dx, dy = -u[1], u[0]
    side = {i: (terms[i][1] - det.cx) * dx + (terms[i][2] - det.cy) * dy
            for i in rest}
    drain, source = sorted(rest, key=lambda i: -side[i])
    note("gate_read")
    return [terms[drain], terms[g], terms[source]]


# --------------------------------------------------------------------------
# The + and - drawn inside a symbol
# --------------------------------------------------------------------------

#: A glyph component may be at most this fraction of the box's area. The
#: triangle outline is far larger; this keeps it out without assuming it is
#: the single largest component, which a fragmented drawing breaks.
MAX_GLYPH_AREA_FRAC = 0.06

#: ...and at least this fraction, so speckle is not read as a minus sign.
MIN_GLYPH_AREA_FRAC = 0.0008

#: ``min_extent / max_extent`` of a glyph's ink. A ``+`` is square by
#: construction; a ``-`` is a single stroke. The gap is wide, so the test is
#: the *difference* between the two sides rather than an absolute cut: both
#: are drawn by the same hand at the same size.
MIN_SHAPE_MARGIN = 0.25


def glyph_blobs(binary: np.ndarray, det, u, info=None):
    """``(across, shape, area, cx, cy)`` for every glyph-sized ink blob.

    Shared by the op-amp inputs and the sources: both mark their terminals
    with a ``+`` and a ``-`` drawn inside the symbol and not touching it,
    so the same search finds both. ``shape`` is ``min/max`` extent -- a
    ``+`` is square by construction, a ``-`` is one stroke.

    Components touching the crop border are dropped: those are the leads and
    whatever ran past the symbol, never a ``+`` drawn inside the triangle.
    """
    h, w = binary.shape[:2]
    x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
    x2, y2 = min(w, int(det.x2)), min(h, int(det.y2))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return []
    crop = (binary[y1:y2, x1:x2] > 0).astype(np.uint8)
    area = crop.shape[0] * crop.shape[1]
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(crop, 8)
    px, py = -u[1], u[0]                       # across the symbol's axis
    out = []
    for i in range(1, n):
        bx, by, bw, bh, a = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                             stats[i, cv2.CC_STAT_AREA])
        if bx == 0 or by == 0 or bx + bw >= crop.shape[1] or by + bh >= crop.shape[0]:
            continue                            # touches the border: not a glyph
        if not (MIN_GLYPH_AREA_FRAC * area <= a <= MAX_GLYPH_AREA_FRAC * area):
            continue
        cx, cy = float(cents[i][0]) + x1, float(cents[i][1]) + y1
        across = (cx - det.cx) * px + (cy - det.cy) * py
        shape = min(bw, bh) / max(bw, bh, 1)
        out.append((across, shape, float(a), cx, cy))
    if info is not None:
        info["glyphs"] = len(out)
    return out




def order_source_terminals(binary, det, terms: list, info=None) -> list:
    """Reorder ``terms`` into SPICE's ``(positive, negative)``.

    A source is drawn as a circle with a ``+`` and a ``-`` inside it, and
    until now nobody read them: the terminals came back in whatever order the
    ring statistics ranked them, so half the sources in any drawing were
    backwards. That is invisible downstream for the same reason a reversed
    diode is -- the netlist parses, LTspice opens it, the round-trip passes,
    and graph isomorphism cannot see direction -- and it is worse than a
    reversed diode, because a supply the wrong way round inverts every node
    voltage that depends on it.

    The glyphs are the same ones an op-amp marks its inputs with, so the same
    :func:`glyph_blobs` search finds them: ink inside the symbol, not touching
    the border, small relative to the box. A ``+`` is compact and a ``-`` is
    one stroke, and the two sides are required to *disagree*, so a stray dash
    beside one terminal cannot be read as a sign on its own.

    Returns the terminals untouched when the drawing does not say -- no two
    terminals, no ink, one glyph, or two of the same shape.
    """
    def note(state, **kw):
        if info is not None:
            info.update(state=state, **kw)

    if binary is None or len(terms) != 2:
        note("not_two_terminals")
        return terms
    u = _unit_vec(terms[1][1] - terms[0][1], terms[1][2] - terms[0][2])
    if u is None:
        note("degenerate_axis")
        return terms

    blobs = glyph_blobs(binary, det, u)
    if len(blobs) < 2:
        note("no_signs")
        return terms

    # Project each glyph onto the terminal-to-terminal axis and keep the
    # largest on each side -- a '+' drawn as two strokes still leaves the
    # longer one, which looks like a '-', so the margin test below is what
    # actually makes the call.
    best: dict = {}
    for _across, shape, area, cx, cy in blobs:
        along = (cx - det.cx) * u[0] + (cy - det.cy) * u[1]
        i = 0 if along < 0 else 1            # terms[0] is at -u, terms[1] at +u
        if i not in best or area > best[i][1]:
            best[i] = (shape, area)
    if len(best) != 2:
        note("one_sided")
        return terms
    s0, s1 = best[0][0], best[1][0]
    if abs(s0 - s1) < MIN_SHAPE_MARGIN:
        note("signs_alike")
        return terms
    note("read")
    plus = 0 if s0 > s1 else 1               # the compact glyph is the '+'
    return [terms[plus], terms[1 - plus]]


def _unit_vec(dx: float, dy: float):
    n = float(np.hypot(dx, dy))
    return None if n < 1e-6 else (dx / n, dy / n)


#: Ink off the lead's own line, as a fraction of the ink on it, before a
#: glyph counts as a '+' rather than a '-'.
MIN_CROSS_STROKE = 0.25


def sign_by_cross_stroke(binary, det, u, terms: list):
    """Which of two leads carries a ``+``, read without isolating the glyph.

    :func:`glyph_blobs` needs the sign to be its own connected component, and
    on a cleanly drawn schematic it is not: the input lead runs through the
    symbol's edge and straight into the ``-``, which is collinear with it, so
    the whole thing is one component touching the crop border and gets
    dropped. Three op-amps of four on one drawing declined for that reason.

    What survives the merge is the *shape*. Both glyphs lie on their lead's
    own line; only the ``+`` puts ink anywhere else. So look just inside the
    symbol, in a band around each lead, and measure how much ink sits off the
    lead's line. A ``-`` leaves almost none, a ``+`` leaves its upright.

    Returns the index of the ``+`` lead, or ``None`` when the two sides do
    not clearly differ.
    """
    if binary is None or len(terms) != 2:
        return None
    h, w = binary.shape[:2]
    px, py = -u[1], u[0]
    half = 0.5 * (abs(u[0]) * det.w + abs(u[1]) * det.h)
    span = 0.5 * (abs(px) * det.w + abs(py) * det.h)
    if half < 6 or span < 6:
        return None
    tol = max(2.0, 0.035 * span)

    scores = []
    for t in terms:
        across0 = (t[1] - det.cx) * px + (t[2] - det.cy) * py
        on = off = 0
        # Step inward from the base edge, which is where both glyphs sit.
        for a in np.linspace(-0.92, -0.35, 24):
            for c in np.linspace(across0 - 0.22 * span, across0 + 0.22 * span, 29):
                x = int(round(det.cx + u[0] * half * a + px * c))
                y = int(round(det.cy + u[1] * half * a + py * c))
                if not (0 <= x < w and 0 <= y < h) or binary[y, x] == 0:
                    continue
                if abs(c - across0) <= tol:
                    on += 1
                else:
                    off += 1
        scores.append((off, on))

    (off0, on0), (off1, on1) = scores
    if on0 < 3 or on1 < 3:
        return None                      # no glyph found on one side at all
    r0, r1 = off0 / max(on0, 1), off1 / max(on1, 1)
    if abs(r0 - r1) < MIN_CROSS_STROKE:
        return None
    return 0 if r0 > r1 else 1
