"""Which lead is the output, and which input is the ``+``? Read the triangle.

``_terminals()`` finds *where* a component's wires leave its box, ranked by how
much skeleton each contact carries. For a resistor that is enough. For an
op-amp it is not, and the failure is worse than a transistor's, because an
op-amp's three pins are not interchangeable in any degree: LTspice writes

    X<ref>  invin  noninvin  out

-- inverting input, non-inverting input, output -- and that order was taken
from the ``SpiceOrder`` attributes in LTspice's own ``OpAmps/opamp.asy`` and
then *confirmed by netlisting a labelled symbol*, because the point is easy to
get backwards:

    SYMBOL OpAmps\\opamp + FLAG AAA/BBB/CCC on the three pins
    -> X§X1 AAA BBB CCC opamp        ;§pnba invin)noninvin)out

``opamp.sub`` agrees: ``.subckt opamp 1 2 3`` with ``G1 0 3 2 1 {Aol}``, a
VCCS driving node 3 from ``V(2) - V(1)``, so 1 is the inverting input. Several
plausible-sounding sources give the order as ``In+ In- Out``; it is not, and
taking their word for it turns an inverting amplifier into positive feedback,
which simulates -- straight into the rail -- without any complaint at all.

Before this module the three nets were written in whatever order
``_ring_label_stats`` happened to rank them, which is by skeleton pixel count:
an ordering with no electrical meaning whatsoever. Nothing downstream could
notice, for the same reason it cannot notice a reversed transistor -- graph
isomorphism over an undirected component/net graph does not see pin roles.

The drawing says it in two steps, and neither is an angle rule.

**The output is the lead on its own.** An op-amp is a triangle: the two inputs
enter the flat base and the output leaves the apex. So of three terminals, two
sit on one side of the symbol and one on the other, and the lone one is the
output. This needs no ink at all and is the rule used whenever three distinct
terminals were found.

**The ink is the fallback, and it is a mass statistic, not an extent one.**
When two pins share a net -- a unity-gain buffer ties the output to the
inverting input, which is the single most common op-amp configuration there is
-- only two distinct labels come back and the lone-terminal rule has nothing to
work with. Then the symbol itself has to say which way it points. The base is
one long stroke spanning the full height while the apex is a point, so ink
*mass* per slice spikes at the base; measured over 121 CGHD op-amps the signed
asymmetry runs 0.2-0.6 the right way. The perpendicular *extent* per slice was
tried first and is much weaker (p10 of 0.21, minimum 0.039) because the
detected box usually catches a neighbour's wire, which spans the box and
flattens the profile.

**The ``+`` and ``-`` are read as separate ink.** Which input is which is not
a geometric fact -- CGHD drafters put the ``+`` on top about as often as on the
bottom -- so it has to come off the page. Both glyphs are drawn *inside* the
triangle and not touching it, so connected components that are neither the
outline nor the border give the candidates, and a ``+`` is compact where a
``-`` is a single elongated stroke. Nothing here distinguishes a ``-`` from a
stray dash, so the two sides are required to disagree: one compact, one
elongated, or the module declines.

Like :mod:`pins`, every step returns the terminals unchanged when the drawing
does not actually say. A known-arbitrary order is better than a confidently
wrong one, and the caller cannot tell them apart afterwards.
"""

from __future__ import annotations

import cv2
import numpy as np

from .pins import (
    MIN_SHAPE_MARGIN, glyph_blobs, lone_lead_direction, sign_by_cross_stroke,
)

#: Terminal roles, in the order LTspice netlists them.
ROLES = ("invin", "noninvin", "out")

#: The lone terminal must clear the other two by this fraction of the symbol's
#: size along the axis before the 2-vs-1 split is believed. A real op-amp puts
#: the output a full symbol-width from its inputs, so this is loose.
MIN_SPLIT_FRAC = 0.25

#: ...and the lone terminal must also sit *between* the other two across the
#: axis, within this fraction of the symbol's width, because the output leaves
#: from the apex and the apex is centred on the base. Without it a transistor-
#: like spread of three leads 120 degrees apart reads as a 2-vs-1 split on
#: whichever axis happens to suit, which is a confident answer to a question
#: the drawing never answered.
MAX_OUT_OFFSET_FRAC = 0.35

#: Signed ink-mass asymmetry needed before the symbol alone decides which end
#: is the base. Measured over 121 CGHD op-amps the correct axis gives a median
#: 0.51 and a 10th percentile of 0.21; below this the crop is usually
#: fragmented or full of a neighbour's wiring, and declining is right.
MIN_MASS_ASYM = 0.20

#: Fraction of the box, from each end, whose ink mass is compared.
MASS_END_FRAC = 0.25

def _unit(dx: float, dy: float):
    n = float(np.hypot(dx, dy))
    return None if n < 1e-6 else (dx / n, dy / n)


def _mass_asymmetry(binary: np.ndarray, det, axis: int) -> float | None:
    """Signed ink-mass asymmetry along ``axis`` (0 = x, 1 = y).

    Positive means the heavy end -- the base -- is at the low end, so the
    apex, and therefore the output, is at the high end.
    """
    h, w = binary.shape[:2]
    x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
    x2, y2 = min(w, int(det.x2)), min(h, int(det.y2))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    crop = binary[y1:y2, x1:x2] > 0
    mass = crop.sum(axis=0 if axis == 0 else 1).astype(float)
    n = mass.size
    k = max(1, int(round(MASS_END_FRAC * n)))
    lo, hi = float(mass[:k].sum()), float(mass[-k:].sum())
    if lo + hi <= 0:
        return None
    return (lo - hi) / (lo + hi)


def _direction_from_ink(binary: np.ndarray, det):
    """Base-to-apex unit vector read from the symbol's own ink, or ``None``."""
    if binary is None:
        return None
    best, axis = 0.0, None
    for a in (0, 1):
        m = _mass_asymmetry(binary, det, a)
        if m is not None and abs(m) > abs(best):
            best, axis = m, a
    if axis is None or abs(best) < MIN_MASS_ASYM:
        return None
    # Positive asymmetry puts the base at the low end, so the apex is at the
    # high end and the direction is positive along that axis.
    s = 1.0 if best > 0 else -1.0
    return (s, 0.0) if axis == 0 else (0.0, s)


def _direction_from_terminals(det, terms: list):
    """Base-to-apex unit vector from a clean 2-vs-1 terminal split.

    The split itself lives in :func:`pins.lone_lead_direction`, because an
    op-amp is not the only symbol built this way: CGHD's own port geometry
    puts a FET's gate at ``[0, 0.5]`` with drain and source at ``[0.5, 0]``
    and ``[0.5, 1]``, and a BJT's base at ``[0, 0.5]`` likewise -- one lead
    alone on one side, two together on the other.
    """
    return lone_lead_direction(det, terms, min_split_frac=MIN_SPLIT_FRAC,
                               max_offset_frac=MAX_OUT_OFFSET_FRAC)


def _plus_from_blobs(binary, det, u, ins, info=None):
    """Index of the ``+`` input from isolated glyphs, or ``None``."""
    blobs = glyph_blobs(binary, det, u, info)
    if len(blobs) < 2:
        return None
    px, py = -u[1], u[0]
    sides = [(t[1] - det.cx) * px + (t[2] - det.cy) * py for t in ins]
    if sides[0] == sides[1]:
        return None
    best: dict = {}
    for across, shape, area, _cx, _cy in blobs:
        i = 0 if abs(across - sides[0]) <= abs(across - sides[1]) else 1
        # Of several blobs on one side keep the largest: a '+' drawn as two
        # separate strokes still leaves the longer one, which is a '-' shape,
        # so this is not sufficient on its own -- the margin test below is.
        if i not in best or area > best[i][1]:
            best[i] = (shape, area)
    if len(best) != 2:
        return None
    s0, s1 = best[0][0], best[1][0]
    if abs(s0 - s1) < MIN_SHAPE_MARGIN:
        return None                             # both the same shape: no call
    if info is not None:
        info["shapes"] = (round(s0, 3), round(s1, 3))
    return 0 if s0 > s1 else 1                  # the compact one is the '+'


def _order_inputs(binary: np.ndarray, det, u, ins: list, info=None):
    """``(inverting, non_inverting)`` read off the drawn signs, or ``None``.

    The sign has to be its own connected component, which is a real limit:
    drawn cleanly, the input lead runs through the symbol's edge and straight
    into the ``-``, collinear with it, so the whole thing is one stroke
    touching the crop border and :func:`glyph_blobs` drops it. Three op-amps
    of four on one rendered schematic decline for that reason.

    **A shape-based fallback was tried and removed.** The idea was sound --
    both glyphs lie on their lead's own line and only the ``+`` puts ink
    anywhere else, so measure ink off that line just inside the symbol. On
    synthetic drawings it worked; on the two real op-amps that needed it, it
    returned the *inverted* answer both times, because the window picks up
    the triangle's own slanted edge rather than the glyph. It turned three
    honest declines into two confident errors and one correct reading, which
    is the trade this module exists to refuse. ``pins.sign_by_cross_stroke``
    is kept, unused, so the measurement is not re-derived from scratch.
    """
    if binary is None or len(ins) != 2:
        return None
    plus = _plus_from_blobs(binary, det, u, ins, info)
    if plus is None:
        return None
    return ins[1 - plus], ins[plus]


def _label_at(labels: np.ndarray, x: float, y: float, radius: int):
    """Most common non-zero node label within ``radius`` of ``(x, y)``."""
    h, w = labels.shape[:2]
    x0, y0 = max(0, int(x) - radius), max(0, int(y) - radius)
    x2, y2 = min(w, int(x) + radius + 1), min(h, int(y) + radius + 1)
    if x2 <= x0 or y2 <= y0:
        return None
    patch = labels[y0:y2, x0:x2]
    vals = patch[patch > 0]
    if vals.size == 0:
        return None
    lab = int(np.bincount(vals.ravel()).argmax())
    ys, xs = np.nonzero(patch == lab)
    return lab, float(xs.mean() + x0), float(ys.mean() + y0)


def _probe_pin(labels: np.ndarray, det, out_dir, across: float):
    """Find the node touching the pin on the ``out_dir`` edge at ``across``.

    ``out_dir`` is the outward normal of the edge the pin leaves by, so the
    output is probed along the base-to-apex vector and the two inputs along
    its negative. ``across`` is the offset along that edge, as a fraction of
    its length, measured from the centre; LTspice's own symbol puts the inputs
    at a quarter and three quarters of the base, which is +-0.5 here.

    The search grows outward because the symbol's box is erased from the node
    map before tracing, so the label does not begin until clear of it.
    """
    px, py = -out_dir[1], out_dir[0]
    half = 0.5 * (abs(out_dir[0]) * det.w + abs(out_dir[1]) * det.h)
    span = 0.5 * (abs(px) * det.w + abs(py) * det.h)
    step = max(3.0, 0.06 * max(det.w, det.h))
    for grow in (1.0, 1.25, 1.6, 2.0):
        x = det.cx + out_dir[0] * half * grow + px * across * span
        y = det.cy + out_dir[1] * half * grow + py * across * span
        got = _label_at(labels, x, y, int(round(step * grow)))
        if got is not None:
            return got
    return None


def order_opamp_terminals(binary, labels, det, terms: list, info=None) -> list:
    """Reorder ``terms`` into LTspice's ``(invin, noninvin, out)``.

    ``binary`` is the ink mask and ``labels`` the per-pixel node map; either
    may be ``None``, in which case the steps that need it are skipped.

    Fills ``info['state']`` with what happened, so the caller can say it out
    loud: a stage that silently declines looks exactly like one that is not
    installed, and both produce a netlist that simulates and is wrong.
    """
    def note(state, **kw):
        if info is not None:
            info.update(state=state, **kw)

    u = _direction_from_terminals(det, terms)
    from_terms = u is not None
    ink = _direction_from_ink(binary, det)
    if u is None:
        u = ink
    elif ink is not None and (ink[0] * u[0] + ink[1] * u[1]) < 0:
        # The terminals and the ink point opposite ways. One of them is wrong
        # and there is nothing here to say which, so make no claim.
        note("direction_conflict")
        return terms
    if u is None:
        note("no_direction")
        return terms
    note("read" if from_terms else "read_from_ink")

    px, py = -u[1], u[0]
    along = [(t[1] - det.cx) * u[0] + (t[2] - det.cy) * u[1] for t in terms]

    if len(terms) == 3:
        out_i = int(np.argmax(along))
        ins = [t for i, t in enumerate(terms) if i != out_i]
        out = terms[out_i]
    elif len(terms) == 2 and labels is not None:
        # Two labels for three pins: a pin is sharing a net with another, which
        # is what a unity-gain buffer looks like. The ring statistics merge the
        # two contacts into one centroid, so neither position is usable and the
        # pins have to be found where the symbol says they are instead.
        back = (-u[0], -u[1])
        got = [_probe_pin(labels, det, u, 0.0),
               _probe_pin(labels, det, back, -0.5),
               _probe_pin(labels, det, back, +0.5)]
        if any(g is None for g in got):
            note("pin_probe_failed")
            return terms
        out, a, b = got
        ins = [a, b]
        note("read_shared" if from_terms else "read_shared_from_ink")
    else:
        note("not_three_terminals")
        return terms

    ordered = _order_inputs(binary, det, u, ins, info)
    if ordered is None:
        # The output is known and the inputs are not. Saying so is worth more
        # than the half-result: an op-amp with its inputs swapped is an
        # oscillator, not a slightly different amplifier.
        note("no_signs")
        return terms
    inv, noninv = ordered
    return [inv, noninv, out]
