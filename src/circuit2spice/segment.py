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
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a uint8 mask, 255 = pencil stroke, at the input's resolution.

    Inference runs at ``long_side`` and the probability map is then resampled
    up. Strokes are only a few pixels wide, so going much below ~1024 starts
    dissolving them; going far above costs time for no gain.
    """
    import torch

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
    return ((prob > threshold) * 255).astype(np.uint8)
