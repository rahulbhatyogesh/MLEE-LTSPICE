"""Wire tracing: photograph -> set of electrical nodes.

This is the stage the reference work skips. Arnadottir & Yngvason (CS230)
restricted themselves to *parallel* circuits precisely because, in their words,
node detection "proved to be a daunting task"; Kumar & Merchant (CS229) never
got past classifying isolated crops. Without this stage a detector gives you a
bill of materials, not a circuit.

The approach:

1. Binarise the photo (CLAHE + adaptive threshold), because CGHD images are
   phone photos of pencil on paper with strongly uneven illumination.
2. Erase every detected *symbol* box. What is left is wires.
3. Skeletonise and take connected components -- each component is a candidate
   electrical node.
4. Repair the two things that breaks:

   - a ``junction`` dot means every wire touching it is the *same* node, so we
     union those fragments;
   - a ``crossover`` means two wires cross *without* connecting, so we erase the
     crossing and re-link only the pairs of stubs that are collinear.

Step 4 is what makes arbitrary (non-parallel) topologies work.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .classes import ERASE_BEFORE_WIRES
from .graph import Detection, UnionFind


@dataclass
class NodeMap:
    """Per-pixel electrical node labels (0 = background, >=1 = node id)."""

    labels: np.ndarray
    n_nodes: int
    skeleton: np.ndarray
    binary: np.ndarray

    def ring_histogram(
        self, box: tuple[float, float, float, float], k: int
    ) -> Counter:
        """Count node labels in the annulus just outside ``box``.

        That annulus is where the wire stubs of a component leave its bounding
        box, so the labels found there are the nodes the component connects to.
        """
        return _ring_histogram(self.labels, box, k)


def binarize(
    bgr: np.ndarray,
    method: str = "flatfield",
    block: int | None = None,
    c: int = 10,
) -> np.ndarray:
    """Return a uint8 mask where 255 = ink stroke, 0 = paper.

    ``flatfield`` (default) divides the image by a heavily blurred copy of
    itself, cancelling the illumination gradient, and then takes a single Otsu
    threshold on the result.

    ``adaptive`` is the textbook CLAHE + adaptive-threshold recipe used by the
    prior work, kept for the ablation in the report.

    The difference matters far more than it looks.  Much of CGHD is drawn on
    *ruled notebook paper*, and adaptive thresholding equalises local contrast
    -- which is precisely what promotes a faint printed rule line to the same
    status as a pencil stroke.  Those rule lines run the width of the page, so
    after skeletonisation they weld every node in the circuit into one, and the
    extracted netlist is garbage no matter how good the detector is.  Flat-field
    division preserves the *absolute* darkness of a stroke, so pencil survives
    and the ruling drops out.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

    if method == "adaptive":
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        g = cv2.medianBlur(g, 3)
        if block is None:
            block = max(11, int(min(g.shape[:2]) * 0.03) | 1)
        binary = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c
        )
    elif method == "unet":
        # Learned stroke/background separation. Returns a clean mask directly,
        # so it skips the morphological clean-up the thresholds need.
        from .segment import segment_strokes

        return segment_strokes(bgr if bgr.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    elif method == "flatfield":
        k = max(3, (min(gray.shape[:2]) // 20) | 1)
        bg = cv2.medianBlur(gray, min(k, 151) | 1)
        norm = cv2.divide(gray, bg, scale=255)
        _, binary = cv2.threshold(
            norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        raise ValueError(f"unknown binarisation method {method!r}")

    # Drop salt noise, then close 1px gaps that thresholding opens in a stroke.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binary


def _clip_rect(
    box: tuple[float, float, float, float], shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(round(x1)), w)),
        max(0, min(int(round(y1)), h)),
        max(0, min(int(round(x2)), w)),
        max(0, min(int(round(y2)), h)),
    )


def _ring_histogram(
    labels: np.ndarray, box: tuple[float, float, float, float], k: int
) -> Counter:
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = _clip_rect((x1 - k, y1 - k, x2 + k, y2 + k), labels.shape)
    if ox2 <= ox1 or oy2 <= oy1:
        return Counter()
    patch = labels[oy1:oy2, ox1:ox2].copy()
    ix1, iy1, ix2, iy2 = _clip_rect(box, labels.shape)
    # Blank the interior so only the surrounding annulus is counted.
    patch[
        max(0, iy1 - oy1) : max(0, iy2 - oy1),
        max(0, ix1 - ox1) : max(0, ix2 - ox1),
    ] = 0
    vals = patch[patch > 0]
    return Counter(vals.tolist())


def _ring_label_angles(
    labels: np.ndarray, box: tuple[float, float, float, float], k: int
) -> dict:
    """Mean angle (radians) of each node label around the centre of ``box``."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ox1, oy1, ox2, oy2 = _clip_rect((x1 - k, y1 - k, x2 + k, y2 + k), labels.shape)
    if ox2 <= ox1 or oy2 <= oy1:
        return {}
    patch = labels[oy1:oy2, ox1:ox2]
    ys, xs = np.nonzero(patch)
    if len(xs) == 0:
        return {}
    vals = patch[ys, xs]
    ang = np.arctan2((ys + oy1) - cy, (xs + ox1) - cx)
    out: dict = {}
    for lab in np.unique(vals):
        sel = vals == lab
        # Circular mean, so angles either side of +/-pi average correctly.
        out[int(lab)] = float(
            np.arctan2(np.sin(ang[sel]).mean(), np.cos(ang[sel]).mean())
        )
    return out


def remove_thin_strokes(
    binary: np.ndarray, q: float = 70.0, floor: float = 1.2
) -> np.ndarray:
    """Erase strokes thinner than the drawing's own typical stroke.

    Printed page ruling is the single worst failure mode on this dataset: it is
    ink-coloured, it runs the width of the page, and once skeletonised it welds
    unrelated nodes together.  Removing it by connected component does not work,
    because wherever a rule line *crosses* a wire the two become one component
    and any component-level rule throws the circuit away with the ruling.

    Thickness separates them pixel-by-pixel instead, so a crossing is cut rather
    than merged.  Printed ruling is 1-2 px wide; a pencil stroke is 3-6 px.  We
    take the distance transform (giving each ink pixel its half-stroke-width),
    keep only pixels above a percentile of that distribution, and dilate the
    surviving cores back to full width.  The threshold is relative to the image
    so it adapts to resolution and to how hard the drafter pressed.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink = dist[binary > 0]
    if ink.size == 0:
        return binary
    t = max(floor, float(np.percentile(ink, q)))
    core = ((dist >= t) * 255).astype(np.uint8)
    k = int(2 * round(t) + 1)
    grown = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = cv2.bitwise_and(grown, binary)
    # If this removed nearly everything the assumption was wrong (e.g. a very
    # lightly drawn circuit), so fall back rather than destroy the image.
    return binary if (out > 0).sum() < 0.15 * (binary > 0).sum() else out


def _skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    """Pixels where a skeleton stroke stops: exactly one 8-neighbour."""
    k = np.ones((3, 3), np.uint8)
    neigh = cv2.filter2D((skel > 0).astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ends = (skel > 0) & (neigh == 2)        # 2 = itself + one neighbour
    ys, xs = np.nonzero(ends)
    return np.stack([xs, ys], axis=1)


def _endpoint_direction(skel: np.ndarray, pt: np.ndarray, walk: int = 7) -> np.ndarray:
    """Unit vector pointing *outward* along the stroke that ends at ``pt``."""
    h, w = skel.shape
    x, y = int(pt[0]), int(pt[1])
    prev = {(x, y)}
    cur = (x, y)
    far = (x, y)
    for _ in range(walk):
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cur[0] + dx, cur[1] + dy
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] and (nx, ny) not in prev:
                    nxt = (nx, ny)
                    break
            if nxt:
                break
        if nxt is None:
            break
        prev.add(nxt)
        cur = far = nxt
    v = np.array([x - far[0], y - far[1]], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, 0.0])


def _segment_hits_box(p, q, boxes) -> bool:
    """True if the segment p-q passes through any erased symbol box."""
    for x1, y1, x2, y2 in boxes:
        # Cheap reject, then sample along the segment.
        if max(p[0], q[0]) < x1 or min(p[0], q[0]) > x2:
            continue
        if max(p[1], q[1]) < y1 or min(p[1], q[1]) > y2:
            continue
        for t in np.linspace(0.0, 1.0, 12):
            sx = p[0] + t * (q[0] - p[0])
            sy = p[1] + t * (q[1] - p[1])
            if x1 <= sx <= x2 and y1 <= sy <= y2:
                return True
    return False


def reconnect_endpoints(
    skel: np.ndarray,
    labels: np.ndarray,
    uf: UnionFind,
    erased_boxes: list,
    *,
    max_gap: float = 18.0,
    min_cos: float = 0.72,
) -> int:
    """Rejoin strokes that thresholding snapped in two.

    This is the counterpart to :func:`remove_thin_strokes`.  Deleting thin
    pixels strips printed ruling, but it also severs a pencil wire wherever the
    drafter eased off and the stroke thinned, which shatters one electrical net
    into several fragments and loses the connection.

    Blind dilation is the wrong repair: it closes a gap by fattening everything,
    so two unrelated wires that merely pass near each other also merge.  Instead
    we look for *stroke ends* -- skeleton pixels with a single neighbour -- and
    rejoin a pair only when the evidence says one stroke continues into the
    other:

    - the two ends belong to different fragments,
    - they are closer than ``max_gap``,
    - each end points at the other (their outward stroke directions are
      roughly anti-parallel, hence ``min_cos``), so a T-shape or two wires
      passing side by side is rejected, and
    - the gap does not cross an erased symbol box -- that gap is *deliberate*,
      and bridging it would short out the component.

    Returns the number of joins made.
    """
    pts = _skeleton_endpoints(skel)
    if len(pts) < 2:
        return 0
    dirs = np.stack([_endpoint_direction(skel, p) for p in pts])
    labs = np.array([labels[p[1], p[0]] for p in pts])

    order = np.argsort(pts[:, 0])
    joins = 0
    used: set = set()
    for ii in range(len(order)):
        i = order[ii]
        if labs[i] == 0 or i in used:
            continue
        best, best_score = None, min_cos
        for jj in range(ii + 1, len(order)):
            j = order[jj]
            if labs[j] == 0 or j in used:
                continue
            d = pts[j] - pts[i]
            if d[0] > max_gap:
                break                       # sorted by x: nothing closer ahead
            dist = float(np.linalg.norm(d))
            if dist < 1e-6 or dist > max_gap:
                continue
            if uf.find(int(labs[i])) == uf.find(int(labs[j])):
                continue
            u = d / dist
            # Each end must point towards the other.
            score = min(float(dirs[i] @ u), float(dirs[j] @ -u))
            if score > best_score and not _segment_hits_box(pts[i], pts[j], erased_boxes):
                best, best_score = j, score
        if best is not None:
            uf.union(int(labs[i]), int(labs[best]))
            used.add(i)
            used.add(best)
            joins += 1
    return joins


def circuit_hull(
    shape: tuple, detections: list, margin_frac: float = 0.10
) -> tuple[int, int, int, int] | None:
    """Bounding box of all detections, expanded by a margin.

    The detector tells us where the circuit *is*; everything well outside that
    region is page furniture.
    """
    boxes = [d for d in detections if d.w > 0 and d.h > 0]
    if not boxes:
        return None
    x1 = min(d.x1 for d in boxes)
    y1 = min(d.y1 for d in boxes)
    x2 = max(d.x2 for d in boxes)
    y2 = max(d.y2 for d in boxes)
    m = margin_frac * max(x2 - x1, y2 - y1)
    return _clip_rect((x1 - m, y1 - m, x2 + m, y2 + m), shape)


def drop_off_circuit(
    binary: np.ndarray,
    hull: tuple | None,
    min_inside: float = 0.5,
    mode: str = "mask",
) -> np.ndarray:
    """Delete strokes that lie mostly outside the circuit.

    Off by default.  It was written to catch surviving rule lines, but measured
    on the 96 annotated photographs it changes nothing once
    :func:`remove_thin_strokes` is in play, and it is actively unsafe when few
    symbols are detected: the hull then hugs the components and clips the very
    wires we are trying to trace.  Kept for the ablation, and because
    ``mode="cc"`` documents the approach that does *not* work -- where a rule
    line crosses a wire the two are one component, so any component-level rule
    discards the circuit along with the ruling.
    """
    if hull is None:
        return binary
    hx1, hy1, hx2, hy2 = hull
    if mode == "mask":
        # Simply erase everything outside the circuit's bounding box.  Safe by
        # construction: it can never remove ink that lies on the circuit.
        out = np.zeros_like(binary)
        out[hy1:hy2, hx1:hx2] = binary[hy1:hy2, hx1:hx2]
        return out
    n, lab, stats, _c = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    mask = np.zeros(binary.shape, dtype=bool)
    mask[hy1:hy2, hx1:hx2] = True
    inside = np.bincount(lab[mask].ravel(), minlength=n).astype(np.float64)
    total = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, inside / total, 0.0)
    drop = np.flatnonzero(frac < min_inside)
    drop = drop[drop != 0]
    if drop.size:
        binary = binary.copy()
        binary[np.isin(lab, drop)] = 0
    return binary


def build_node_map(
    bgr: np.ndarray,
    detections: list,
    *,
    min_fragment_px: int = 12,
    bridge_px: int | None = 6,
    method: str = "flatfield",
    drop_outside: str = "none",
    thin_q: float | None = 50.0,
    reconnect_px: float = 24.0,
    binary: np.ndarray | None = None,
) -> NodeMap:
    """Trace wires and resolve them into electrical nodes."""
    if binary is None:
        binary = binarize(bgr, method=method)
    # 0. Strip page ruling and margin furniture before anything else.
    #    The thickness filter exists to remove printed ruling a threshold
    #    cannot; the learned segmenter already excludes it, and applying the
    #    filter on top would only sever thin pencil strokes.
    if thin_q is not None and method != "unet":
        binary = remove_thin_strokes(binary, thin_q)
    if drop_outside and drop_outside != "none":
        binary = drop_off_circuit(
            binary, circuit_hull(binary.shape, detections), mode=drop_outside
        )
    work = binary.copy()

    # 1. Erase symbols and text. Junctions stay (they carry connectivity);
    #    crossovers are erased so the two crossing wires come apart.
    for d in detections:
        if d.cls in ERASE_BEFORE_WIRES or d.cls == "crossover":
            x1, y1, x2, y2 = _clip_rect(d.box, work.shape)
            work[y1:y2, x1:x2] = 0

    # 2. Skeletonise what remains, and label the fragments.
    #
    #    A pencil line photographed under uneven light does not threshold into
    #    one unbroken stroke; it breaks into dozens of pieces.  Labelling the
    #    skeleton directly would then split a single electrical node across
    #    many fragments and lose connections.  So we label a *dilated* copy --
    #    which closes those gaps -- and transfer the labels back onto the
    #    skeleton.  The bridge radius scales with image size and is the one
    #    knob trading missed connections against falsely merged nodes.
    skel = skeletonize(work > 0).astype(np.uint8)
    if bridge_px is None:
        bridge_px = max(2, int(round(min(work.shape[:2]) / 320)))
    if bridge_px > 0:
        ksz = 2 * bridge_px + 1
        bridged = cv2.dilate(skel, np.ones((ksz, ksz), np.uint8))
        n_lab, lab_d = cv2.connectedComponents(bridged, connectivity=8)
        labels = lab_d * skel
    else:
        n_lab, labels = cv2.connectedComponents(skel, connectivity=8)

    # 3. Discard specks: they are threshold noise, not wires.
    if n_lab > 1:
        counts = np.bincount(labels.ravel(), minlength=n_lab)
        tiny = np.flatnonzero(counts < min_fragment_px)
        tiny = tiny[tiny != 0]
        if tiny.size:
            labels[np.isin(labels, tiny)] = 0

    uf = UnionFind()
    for lab in range(1, n_lab):
        uf.find(lab)

    # 4a. A junction dot ties together every fragment that touches it.
    for d in detections:
        if d.cls != "junction":
            continue
        k = max(4, int(0.6 * max(d.w, d.h)))
        hits = [
            lab for lab, cnt in _ring_histogram(labels, d.box, k).items() if cnt >= 2
        ]
        for other in hits[1:]:
            uf.union(hits[0], other)

    # 4b. A crossover must re-link only collinear stubs: left<->right and
    #     top<->bottom, never left<->top.
    for d in detections:
        if d.cls != "crossover":
            continue
        k = max(5, int(0.5 * max(d.w, d.h)))
        angles = _ring_label_angles(labels, d.dilated(k), k)
        items = list(angles.items())
        used: set = set()
        for i, (la, aa) in enumerate(items):
            if la in used:
                continue
            best, best_err = None, np.pi / 4  # tolerance: 45 degrees
            for lb, ab in items[i + 1 :]:
                if lb in used:
                    continue
                # How far from exactly antipodal are these two stubs?
                err = abs(abs((aa - ab + np.pi) % (2 * np.pi) - np.pi) - np.pi)
                if err < best_err:
                    best, best_err = lb, err
            if best is not None:
                uf.union(la, best)
                used.add(la)
                used.add(best)

    # 4c. Rejoin strokes that thresholding snapped in two.  Runs last so the
    #     explicit junction/crossover annotations take priority and this only
    #     repairs what is still broken.
    if reconnect_px:
        erased = [
            _clip_rect(d.box, work.shape)
            for d in detections
            if d.cls in ERASE_BEFORE_WIRES or d.cls == "crossover"
        ]
        reconnect_endpoints(skel, labels, uf, erased, max_gap=float(reconnect_px))

    # 5. Compact the merged labels into a dense 1..N node numbering.
    remap = np.zeros(max(n_lab, 1), dtype=np.int32)
    next_id = 0
    for lab in range(1, n_lab):
        root = uf.find(lab)
        if remap[root] == 0:
            next_id += 1
            remap[root] = next_id
        remap[lab] = remap[root]
    final = remap[labels] if n_lab > 1 else labels

    return NodeMap(labels=final, n_nodes=next_id, skeleton=skel, binary=binary)


def colorize(node_map: NodeMap, seed: int = 0) -> np.ndarray:
    """Render the node map with one colour per node (used as a report figure)."""
    rng = np.random.default_rng(seed)
    lut = rng.integers(60, 255, size=(node_map.n_nodes + 1, 3), dtype=np.uint8)
    lut[0] = 0
    out = lut[node_map.labels]
    return cv2.dilate(out, np.ones((2, 2), np.uint8))
