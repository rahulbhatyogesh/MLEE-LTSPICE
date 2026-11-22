# -*- coding: utf-8 -*-
"""Which way round a diode is drawn.

Four hand-built ink statistics were measured against 445 annotated CGHD
diodes and the best reached 76% (see CLAUDE.md). The reason is structural --
a diode carries a wide perpendicular stroke at *both* ends, the bar at one
and the triangle's base at the other, and the apex abuts the bar, so there is
no gap to measure. That is a shape judgement, not a threshold, so it is
learned.

**The task is posed as angle regression, not as a bit.** CGHD ships a
``<rotation>`` on every symbol and it is continuous -- 285 distinct values
over the 5145 labelled diodes, clustered near the axes but with a real spread
and a diagonal population. Asking for the angle rather than for "which end"
buys three things a binary head cannot have:

* **Rotation and reflection become exact augmentation.** Turn the crop by a
  and the answer turns by -a; mirror it and the answer mirrors. The label
  follows in closed form, so the whole of O(2) is free training data. That is
  what makes 5145 crops enough.
* **No canonicalisation.** The previous attempt rotated each crop onto the
  ink's own principal axis, which is set by whatever is in the box -- leads,
  and routinely the label written beside the part. Here the model is handed
  the crop as it lies.
* **A confidence that means something.** The pipeline already knows the
  diode's axis, because it knows its two terminals. The angle the model
  predicts is an *independent* estimate of that same line, so how well the
  two agree is evidence, and the answer can be declined when they do not.

**The angle is not CGHD's ``<rotation>``; it is derived from it.** The
dataset's own ``classes_ports.json`` puts a diode's anode at unit ``x=0`` and
its cathode at ``x=1``, and its ``utils.py`` maps unit space into the image by
flipping y and *then* rotating by ``-rotation``. Composing the two, the anode
lies in image-coordinate direction ``180 - rotation``. That is not the same as
using ``rotation`` directly, and the difference is invisible on a vertical
diode -- the two agree exactly at 90 and 270 degrees -- which is how the first
attempt came to use the raw angle, verify it by eye on drafter 1 (whose diodes
are nearly all vertical) and then score at chance on everything else.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np

#: The crop the model sees. ResNet's stem divides by 32, so 96 leaves a 3x3
#: feature map -- enough spatial detail to survive the head.
SIZE = 96

#: Square crop side, as a multiple of the box's longer edge. The leads carry
#: no polarity, but the margin is what lets a rotated crop stay filled.
CROP_SCALE = 1.45

#: Wire tracing runs at this resolution and so does the crop, so that the
#: strokes the model is trained on are as thick as the ones it will be shown.
WORK_MAX_DIM = 1600

#: Below this the model says nothing and the caller keeps the order it had.
#: A reversed diode is a worse answer than an unordered one.
MIN_CONFIDENCE = 0.80


# --------------------------------------------------------------------------
# the crop
# --------------------------------------------------------------------------

def _stretch(gray: np.ndarray) -> np.ndarray:
    """Percentile contrast stretch.

    32 drafters means 32 pens, papers and phone cameras. The shape is the
    signal and the exposure is not, so the crop is normalised before the
    model ever sees it rather than left for the first convolution to undo.
    """
    lo, hi = np.percentile(gray, (2.0, 98.0))
    if hi - lo < 1e-3:
        return np.zeros_like(gray, np.uint8)
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_symbol(bgr: np.ndarray, box, size: int = SIZE):
    """A square, contrast-normalised, ink-bright crop around ``box``.

    ``box`` is ``(x1, y1, x2, y2)`` in the coordinates of ``bgr``. The crop is
    square so that rotating it is closed -- a rectangle rotated by 90 degrees
    is a different picture of the same thing, and the model would otherwise
    have to learn that for nothing.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    side = CROP_SCALE * max(x2 - x1, y2 - y1)
    if side < 4:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = int(round(side / 2))
    # Reflect rather than pad with a constant: a flat border is a strong edge
    # the convolutions would read as a stroke.
    pad = half + 2
    big = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    ix, iy = int(round(cx)) + pad, int(round(cy)) + pad
    sub = big[iy - half:iy + half, ix - half:ix + half]
    if sub.size == 0 or sub.shape[0] < 4 or sub.shape[1] < 4:
        return None
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    # Ink bright. Nothing downstream depends on the sign, but a consistent one
    # means the augmentation's brightness jitter moves the paper, not the ink.
    return 255 - _stretch(gray)


def downscale(bgr: np.ndarray, max_dim: int = WORK_MAX_DIM):
    """Match the pipeline's working resolution. Returns ``(image, scale)``."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return bgr, s


# --------------------------------------------------------------------------
# augmentation -- the point of the whole formulation
# --------------------------------------------------------------------------

def rotate_crop(img: np.ndarray, deg: float, scale: float = 1.0,
                dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """Rotate about the centre. A ``cv2`` angle of ``a`` sends a target ``t``
    to ``t - a``; there is a test pinning exactly that."""
    n = img.shape[0]
    m = cv2.getRotationMatrix2D((n / 2.0 - 0.5, n / 2.0 - 0.5), deg, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(img, m, (n, n), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def augment(img: np.ndarray, theta: float, rng):
    """One sample's worth of O(2), plus photometrics.

    ``theta`` is in degrees, in image coordinates. The group acts on the label
    in closed form, which is the whole reason the angle is regressed rather
    than the bit classified:

    * ``cv2`` rotation by ``a``  ->  ``theta - a``
    * mirror left-right          ->  ``180 - theta``
    * mirror top-bottom          ->  ``-theta``
    """
    if rng.random() < 0.5:
        img, theta = img[:, ::-1].copy(), 180.0 - theta
    if rng.random() < 0.5:
        img, theta = img[::-1, :].copy(), -theta
    a = float(rng.uniform(0.0, 360.0))
    img = rotate_crop(img, a, scale=float(rng.uniform(0.85, 1.18)),
                      dx=float(rng.uniform(-4, 4)), dy=float(rng.uniform(-4, 4)))
    theta -= a

    f = img.astype(np.float32)
    f *= float(rng.uniform(0.75, 1.3))                 # contrast
    f += float(rng.uniform(-25, 25))                   # brightness
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        f = cv2.GaussianBlur(f, (k, k), 0)
    if rng.random() < 0.3:
        f = f + rng.normal(0, float(rng.uniform(3, 12)), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8), float(theta % 360.0)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

def build_model(pretrained: bool = True):
    """ResNet-18 with a flattening head, predicting ``(cos t, sin t)``.

    Two deliberate choices:

    * **ImageNet weights.** There is no public hand-drawn-diode model to start
      from, and a stroke drawing is not ImageNet, but the early filters are
      edges and corners either way. The gain over scratch is measured rather
      than assumed -- see ``outputs/diode_orient.json``.
    * **The head flattens; it does not average.** Global average pooling is
      translation-invariant, and where the bar sits relative to the triangle
      is precisely the question. The previous attempt could not fit its own
      training data until this stopped being a mean.
    """
    import torch.nn as nn
    import torchvision

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.avgpool = nn.Identity()
    g = SIZE // 32
    net.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.2),
                           nn.Linear(512 * g * g, 2))
    return net


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

_CACHE: dict = {}


def _load(model_path=None):
    """The TorchScript model, or ``None`` -- including when it does not fit.

    The probe forward pass is the point. A model file left over from an older
    crop shape loads perfectly well and then raises on the first real crop,
    deep inside the pipeline, where the only sign is a traceback out of a
    stage that is meant to be optional. Running one zero crop through it here
    turns that into the same "no model" path as a missing file.
    """
    if "m" in _CACHE:
        return _CACHE["m"]
    path = Path(model_path or os.environ.get("CIRCUIT2SPICE_DIODE")
                or Path(__file__).resolve().parents[2] / "models" / "diode_orient.ts.pt")
    _CACHE["m"] = None
    try:
        import torch

        model = torch.jit.load(str(path), map_location="cpu").eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, SIZE, SIZE))
        if tuple(out.shape) != (1, 2):
            raise ValueError(f"expected a (1, 2) bearing, got {tuple(out.shape)}")
        _CACHE["m"] = model
    except Exception:
        _CACHE["m"] = None
    return _CACHE["m"]


#: Test-time augmentation. The crop is shown at four right angles and each
#: prediction is turned back before they are averaged, so the resultant's
#: *length* is a free agreement measure: four views that disagree cancel.
_TTA = (0.0, 90.0, 180.0, 270.0)


def combine_tta(vectors, angles=_TTA):
    """Fold predictions made on rotated views back into one bearing.

    ``vectors`` are the raw ``(cos, sin)`` outputs for the same crop rotated
    by each of ``angles``. A view turned by ``+a`` has had its target turned
    to ``theta - a``, so its answer is turned back by ``+a`` before averaging.

    Returns ``(bearing_degrees, agreement)``, agreement being the length of
    the mean unit vector: four views that disagree cancel, and that is the
    confidence. Training and inference both come through here, so the two
    cannot drift apart -- a sign slip in one of two copies of this would be
    invisible until the pipeline quietly reversed every diode.
    """
    acc = np.zeros(2, np.float64)
    for (vx, vy), a in zip(vectors, angles):
        n = math.hypot(float(vx), float(vy))
        if n < 1e-9:
            continue
        t = math.atan2(float(vy) / n, float(vx) / n) + math.radians(a)
        acc += (math.cos(t), math.sin(t))
    acc /= len(angles)
    agree = float(math.hypot(acc[0], acc[1]))
    if agree < 1e-9:
        return None
    return math.degrees(math.atan2(acc[1], acc[0])) % 360.0, agree


def predict_direction(bgr: np.ndarray, box, model_path=None):
    """Unit vector pointing from the diode's cathode toward its anode.

    Returns ``(vx, vy, agreement)`` in image coordinates, or ``None`` when
    there is no model or no crop. ``agreement`` is the length of the averaged
    test-time-augmented unit vectors, in ``[0, 1]``.
    """
    model = _load(model_path)
    if model is None:
        return None
    crop = crop_symbol(bgr, box)
    if crop is None:
        return None

    import torch

    views = np.stack([rotate_crop(crop, a) for a in _TTA])
    with torch.no_grad():
        x = torch.from_numpy(views).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
        out = model(x).numpy()
    got = combine_tta(out)
    if got is None:
        return None
    bearing, agree = got
    r = math.radians(bearing)
    return math.cos(r), math.sin(r), agree


def model_available(model_path=None) -> bool:
    """Whether a usable model is installed. Cheap after the first call."""
    return _load(model_path) is not None


def order_diode_terminals(bgr: np.ndarray, det, terms: list,
                          min_confidence: float = MIN_CONFIDENCE,
                          info: dict | None = None) -> list:
    """Reorder ``terms`` into SPICE's ``(anode, cathode)``.

    Returns the input unchanged whenever the evidence is not there: no model,
    not two terminals, a reading the model is unsure of, or a reading that
    does not line up with the axis the terminals themselves define. That last
    test is the one worth having -- the terminals are known independently, so
    a predicted angle lying across them rather than along them is a statement
    about some other ink, and is thrown away.

    ``info``, if given, is filled with ``state`` -- one of ``read``,
    ``no_model``, ``not_two_terminals``, ``no_reading``, ``degenerate_axis``
    or ``low_confidence`` -- and with ``confidence`` when there was one. The
    caller needs it to *say* what happened: a stage that silently declines is
    indistinguishable from one that is not installed, which is the trap this
    project has already paid for twice (see CLAUDE.md on LTspice and on the
    Ollama reader returning empty strings).
    """
    def note(state, **kw):
        if info is not None:
            info["state"] = state
            info.update(kw)

    if bgr is None or len(terms) != 2:
        note("not_two_terminals")
        return terms
    if not model_available():
        note("no_model")
        return terms
    got = predict_direction(bgr, (det.x1, det.y1, det.x2, det.y2))
    if got is None:
        note("no_reading")
        return terms
    vx, vy, agree = got

    ax, ay = terms[1][1] - terms[0][1], terms[1][2] - terms[0][2]
    n = math.hypot(float(ax), float(ay))
    if n < 1e-6:
        note("degenerate_axis")
        return terms
    ax, ay = ax / n, ay / n
    along = vx * ax + vy * ay                 # cosine of the angle between them
    # Confidence is the two pieces of evidence together: how firmly the model
    # answered, and how well its answer lies along the line the terminals
    # already draw.
    confidence = agree * abs(along)
    if confidence < min_confidence:
        note("low_confidence", confidence=confidence)
        return terms
    note("read", confidence=confidence)
    # The vector points at the anode. A positive `along` means it points from
    # terms[0] toward terms[1], so terms[1] is the anode and leads.
    return [terms[1], terms[0]] if along > 0 else list(terms)
