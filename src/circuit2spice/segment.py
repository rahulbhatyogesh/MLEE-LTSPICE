"""Learned stroke segmentation — the replacement for thresholding.

Everything upstream of wire tracing used to be hand-tuned: CLAHE or flat-field
division, an Otsu threshold, a stroke-thickness filter, a hull filter. Each was
added to patch a failure the previous one caused, and the root problem never
went away — a threshold cannot tell a faint printed rule line from a light
pencil stroke, because locally they look identical.

CGHD ships pixel-level stroke masks. A U-Net trained on them answers the
question directly: *is this pixel pencil, or is it paper?* It has seen ruled
notebook paper and learned that the printed lines are not ink, which no
threshold can express.

The model is TorchScript, so this module needs only ``torch`` — not the
``segmentation_models_pytorch`` stack it was trained with.

    from circuit2spice.wires import build_node_map
    build_node_map(img, dets, method="unet")
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

#: Where to look for the exported model, in order.
SEARCH = (
    os.environ.get("CIRCUIT2SPICE_UNET", ""),
    "models/unet_strokes.ts.pt",
    "runs/unet_strokes.ts.pt",
    "unet_strokes.ts.pt",
)

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

#: Probability above which a pixel counts as stroke.
#:
#: **Not 0.5, and that is the whole point.** CGHD's masks do not agree about
#: printed page ruling: sampling the training drafters, some annotators erased
#: the rules and some kept them. The network learns both, so on a ruled page it
#: reports *uncertainty* over the rules -- high enough to clear 0.5, far below
#: the near-1.0 it gives a pen stroke. The threshold is what cashes that
#: uncertainty in, and it is the one knob that turns a model which reproduces
#: the ruling into one that removes it.
#:
#: Tuned end-to-end on drafter 1: 0.500 exact match at 0.5, 0.778 at 0.995,
#: collapsing past 0.999 as strokes themselves start to erode. The optimum is
#: a plateau (0.71-0.78 across 0.992-0.997), not a spike, but it *is* tuned on
#: the same images we report, because drafter 1 ships the only reference
#: schematics in CGHD and so is the only end-to-end signal that exists.
DEFAULT_THRESHOLD = float(os.environ.get("CIRCUIT2SPICE_UNET_THR", "0.995"))

#: Optional second, lower cut-off: keep a pixel if the network is confident
#: (above :data:`DEFAULT_THRESHOLD`) **or** if it is merely plausible and
#: belongs to a thick stroke. The idea is that ruling is thin *and* uncertain
#: while a faint pencil line is thin but part of a stroke that is thick
#: somewhere, so thickness ought to rescue the faint ink the high cut-off
#: erodes.
#:
#: **Off, because it was measured and it loses.** It does what it was built
#: for -- C8, the densest drawing, goes 4/8 to 5/8 -- but every setting costs
#: more elsewhere than it gains: best hybrid 0.764 against 0.778 for the plain
#: threshold, and the loss grows monotonically with how much of the uncertain
#: band is let back in (`outputs/ablation_probthin.json`). Kept because the C8
#: result shows the signal is real, so a per-image rule could still use it.
DEFAULT_LOW_THRESHOLD = (float(os.environ["CIRCUIT2SPICE_UNET_LOW"])
                         if os.environ.get("CIRCUIT2SPICE_UNET_LOW") else None)
DEFAULT_THIN_Q = float(os.environ.get("CIRCUIT2SPICE_UNET_THIN_Q", "70"))


class StrokeModelMissing(FileNotFoundError):
    pass


def find_model(path: str | Path | None = None) -> Path:
    cands = [path] if path else list(SEARCH)
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    raise StrokeModelMissing(
        "no stroke-segmentation model found. Train it with "
        "colab/train_unet_strokes.ipynb, then put unet_strokes.ts.pt in models/ "
        "or set CIRCUIT2SPICE_UNET to its path."
    )


@lru_cache(maxsize=2)
def _load(path: str):
    import torch

    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model


def available(path: str | Path | None = None) -> bool:
    try:
        find_model(path)
    except StrokeModelMissing:
        return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def segment_strokes(
    bgr: np.ndarray,
    model_path: str | Path | None = None,
    *,
    long_side: int = 1024,
    threshold: float | None = None,
    low_threshold: float | None = None,
    thin_q: float | None = None,
) -> np.ndarray:
    """Return a uint8 mask, 255 = pencil stroke, at the input's resolution.

    Inference runs at ``long_side`` and the probability map is then resampled
    up. Strokes are only a few pixels wide, so going much below ~1024 starts
    dissolving them; going far above costs time for no gain.

    ``threshold`` defaults to :data:`DEFAULT_THRESHOLD`, which is nowhere near
    0.5 -- see the note there before changing it.
    """
    import torch

    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    if low_threshold is None:
        low_threshold = DEFAULT_LOW_THRESHOLD
    if thin_q is None:
        thin_q = DEFAULT_THIN_Q

    model = _load(str(find_model(model_path)))

    h, w = bgr.shape[:2]
    scale = long_side / max(h, w)
    # The network downsamples by 32, so both sides must be a multiple of it.
    nw = max(32, int(round(w * scale)) // 32 * 32)
    nh = max(32, int(round(h * scale)) // 32 * 32)
    small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (rgb - _MEAN) / _STD
    tensor = torch.from_numpy(x).permute(2, 0, 1)[None]

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = ((prob > threshold) * 255).astype(np.uint8)
    if low_threshold is not None and low_threshold < threshold:
        from .wires import remove_thin_strokes

        plausible = ((prob > low_threshold) * 255).astype(np.uint8)
        mask = cv2.bitwise_or(mask, remove_thin_strokes(plausible, thin_q))
    return mask
