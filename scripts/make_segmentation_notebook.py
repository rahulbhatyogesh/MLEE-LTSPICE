"""Generate colab/train_unet_strokes.ipynb.

Written as a generator rather than by hand so the cell sources can contain
backslashes and quotes without fighting shell escaping.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS = [
("markdown", """# CGHD stroke segmentation — replace thresholding with a learned model

Trains a U-Net to answer one question per pixel: **pencil, or paper?**

This replaces `binarize()`, `remove_thin_strokes()` and the thickness
threshold — four hand-tuned heuristics collapse into one learned function that
has actually seen ruled notebook paper and knows the printed lines are not ink.

**Drafter 1 is excluded from training.** Its 17 masks are held out because
drafter 1 supplies the only reference schematics in CGHD and is therefore the
entire end-to-end benchmark. Training on it would invalidate every number.

Runtime → Change runtime type → **T4 GPU**. Expect 25–40 min."""),

("code", """# Cell 1 — setup
!nvidia-smi --query-gpu=name,memory.total --format=csv
%pip install -q segmentation-models-pytorch==0.3.4 huggingface_hub
from google.colab import drive
drive.mount('/content/drive')"""),

("code", '''# Cell 2 — fetch ONLY the (image, mask) pairs; drafter_1 held out
import os, collections
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "lowercaseonly/cghd"
files = list_repo_files(REPO, repo_type="dataset")

masks = [f for f in files if "/segmentation/" in f]
images = [f for f in files if "/images/" in f]
# stem -> image path, so a mask can find its photo whatever the extension is
by_stem = {}
for f in images:
    by_stem[(f.split("/")[0], f.rsplit("/", 1)[-1].rsplit(".", 1)[0])] = f

HELD_OUT = "drafter_1"          # the end-to-end benchmark drafter
pairs, skipped, held = [], 0, 0
for m in masks:
    d = m.split("/")[0]
    stem = m.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    img = by_stem.get((d, stem))
    if img is None:
        skipped += 1
    elif d == HELD_OUT:
        held += 1
    else:
        pairs.append((img, m))

print(f"{len(pairs)} training pairs | {held} held out ({HELD_OUT}) | {skipped} masks with no image")

def grab(p):
    return hf_hub_download(REPO, p, repo_type="dataset", local_dir="/content/seg")

with ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(grab, [p for pair in pairs for p in pair]))
print("downloaded")
!du -sh /content/seg'''),

("code", '''# Cell 3 — check mask polarity before trusting it
import numpy as np, cv2, matplotlib.pyplot as plt

ip, mp = pairs[0]
img = cv2.cvtColor(cv2.imread(f"/content/seg/{ip}"), cv2.COLOR_BGR2RGB)
msk = cv2.imread(f"/content/seg/{mp}", cv2.IMREAD_GRAYSCALE)
print("image", img.shape, "| mask", msk.shape, "| unique-ish:", np.unique(msk)[:6], "...")

frac_white = (msk > 127).mean()
print(f"fraction of mask that is white: {frac_white:.3f}")
STROKE_IS_WHITE = frac_white < 0.5     # ink covers a small part of the page
print("=> stroke is", "WHITE" if STROKE_IS_WHITE else "BLACK", "in these masks")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img); ax[0].set_title("photo")
ax[1].imshow(msk, cmap="gray"); ax[1].set_title("mask as stored")
ax[2].imshow((msk > 127) if STROKE_IS_WHITE else (msk <= 127), cmap="gray")
ax[2].set_title("interpreted as stroke=1")
[a.axis("off") for a in ax]; plt.tight_layout()'''),

("code", '''# Cell 4 — dataset
import random, torch
from torch.utils.data import Dataset, DataLoader

CROP = 384

class Strokes(Dataset):
    """Random crops at native resolution. Strokes are a few pixels wide, so
    downscaling the whole page would destroy the very thing we are learning."""

    def __init__(self, pairs, train=True, crop=CROP):
        self.pairs, self.train, self.crop = pairs, train, crop

    def __len__(self):
        return len(self.pairs) * (8 if self.train else 1)

    def __getitem__(self, i):
        ip, mp = self.pairs[i % len(self.pairs)]
        img = cv2.imread(f"/content/seg/{ip}")
        msk = cv2.imread(f"/content/seg/{mp}", cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            return self[(i + 1) % len(self)]
        if msk.shape[:2] != img.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = (msk > 127) if STROKE_IS_WHITE else (msk <= 127)
        m = m.astype(np.float32)

        c = self.crop
        H, W = img.shape[:2]
        if H < c or W < c:
            img = cv2.copyMakeBorder(img, 0, max(0, c - H), 0, max(0, c - W), cv2.BORDER_REFLECT)
            m = cv2.copyMakeBorder(m, 0, max(0, c - H), 0, max(0, c - W), cv2.BORDER_REFLECT)
            H, W = img.shape[:2]

        if self.train:
            # Prefer crops that actually contain ink: most of a page is blank.
            for _ in range(8):
                y, x = random.randint(0, H - c), random.randint(0, W - c)
                if m[y:y+c, x:x+c].mean() > 0.01:
                    break
        else:
            y, x = (H - c) // 2, (W - c) // 2
        img, m = img[y:y+c, x:x+c], m[y:y+c, x:x+c]

        if self.train:
            if random.random() < 0.5: img, m = img[:, ::-1], m[:, ::-1]
            if random.random() < 0.5: img, m = img[::-1], m[::-1]
            k = random.randint(0, 3)
            if k: img, m = np.rot90(img, k), np.rot90(m, k)
            # Lighting jitter: the whole point is robustness to phone photos.
            img = np.clip(img.astype(np.float32) * random.uniform(0.65, 1.35)
                          + random.uniform(-35, 35), 0, 255)

        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return (torch.from_numpy(img).permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(m))[None])

random.Random(0).shuffle(pairs)
n_val = max(8, len(pairs) // 10)
val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
print(f"train {len(train_pairs)} masks, val {len(val_pairs)} masks")

train_dl = DataLoader(Strokes(train_pairs, True), batch_size=8, shuffle=True,
                      num_workers=2, drop_last=True, persistent_workers=True)
val_dl = DataLoader(Strokes(val_pairs, False), batch_size=8, num_workers=2)'''),

("code", '''# Cell 5 — train
import segmentation_models_pytorch as smp

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)

dice = smp.losses.DiceLoss(mode="binary")
bce = torch.nn.BCEWithLogitsLoss()
loss_fn = lambda p, y: 0.5 * bce(p, y) + 0.5 * dice(p, y)

EPOCHS = 30
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, epochs=EPOCHS, steps_per_epoch=len(train_dl))
scaler = torch.cuda.amp.GradScaler()

def iou(logits, y, thr=0.5):
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * y).sum((1, 2, 3))
    union = ((p + y) > 0).float().sum((1, 2, 3))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()

best = 0.0
for ep in range(1, EPOCHS + 1):
    model.train(); tot = 0.0
    for x, y in train_dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()
    model.eval(); vi = []
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(dev), y.to(dev)
            with torch.cuda.amp.autocast():
                vi.append(iou(model(x), y))
    v = float(np.mean(vi))
    flag = ""
    if v > best:
        best = v
        torch.save(model.state_dict(), "/content/unet_best_state.pt")
        flag = "  <- best"
    print(f"epoch {ep:2}/{EPOCHS}  loss {tot/len(train_dl):.4f}  val IoU {v:.4f}{flag}")

print(f"\\nbest val IoU: {best:.4f}")'''),

("code", '''# Cell 6 — look at it before trusting it
model.load_state_dict(torch.load("/content/unet_best_state.pt"))
model.eval()

ip, mp = val_pairs[0]
img = cv2.cvtColor(cv2.imread(f"/content/seg/{ip}"), cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
s = 1024 / max(H, W)
rz = cv2.resize(img, (int(W*s)//32*32, int(H*s)//32*32))
x = (rz.astype(np.float32)/255.0 - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225], np.float32)
with torch.no_grad():
    pr = torch.sigmoid(model(torch.from_numpy(x).permute(2,0,1)[None].to(dev)))[0,0].cpu().numpy()

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(rz); ax[0].set_title("photo")
ax[1].imshow(pr, cmap="magma"); ax[1].set_title("predicted stroke probability")
ax[2].imshow(pr > 0.5, cmap="gray"); ax[2].set_title("thresholded at 0.5")
[a.axis("off") for a in ax]; plt.tight_layout()'''),

("code", '''# Cell 7 — export TorchScript so local inference needs only torch
model.eval().cpu()
traced = torch.jit.trace(model, torch.randn(1, 3, 384, 384))
traced = torch.jit.freeze(traced)
out = "/content/drive/MyDrive/mlee_cghd/unet_strokes.ts.pt"
os.makedirs(os.path.dirname(out), exist_ok=True)
traced.save(out)
print("saved", out, os.path.getsize(out)/1e6, "MB")

# sanity: reload exactly as the local pipeline will
rl = torch.jit.load(out)
with torch.no_grad():
    o = rl(torch.randn(1, 3, 384, 384))
print("reload OK, output", tuple(o.shape))

from google.colab import files
files.download(out)'''),
]


def main() -> None:
    nb = {
        "cells": [
            {
                "cell_type": t,
                "metadata": {},
                **(
                    {"source": s.splitlines(keepends=True)}
                    if t == "markdown"
                    else {"source": s.splitlines(keepends=True),
                          "outputs": [], "execution_count": None}
                ),
            }
            for t, s in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    p = Path("colab/train_unet_strokes.ipynb")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    bad = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(src.lstrip().startswith(m) or ("\n" + m) in src for m in ("!", "%")):
            print(f"  cell {i}: contains magics, skipped syntax check")
            continue
        try:
            compile(src, f"cell{i}", "exec")
            print(f"  cell {i}: compiles OK")
        except SyntaxError as e:
            bad.append((i, e))
            print(f"  cell {i}: SYNTAX ERROR {e}")
    print(f"\nwrote {p.resolve()} ({p.stat().st_size} bytes)")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
