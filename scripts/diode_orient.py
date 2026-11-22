# -*- coding: utf-8 -*-
"""Build, cross-validate and export the diode orientation model.

A diode's polarity is the one thing the pipeline could not read off the page.
``pins.py`` reads a transistor's C/B/E at 1.000 because the bar and the
arrowhead are separate marks with a clear margin between them; a diode has a
wide perpendicular stroke at *both* ends, so no ink statistic separates them
(four were measured, the best reached 76%). This learns it instead.

CGHD ships a ``<rotation>`` on every symbol: 5145 labelled diodes across 32
drafters, which is the whole supply and 10x what the first attempt used. The
angle is regressed rather than the bit classified, because that makes the
rotation and reflection group an exact augmentation -- see ``diode_net``.

    python scripts/diode_orient.py --build        # crops -> data/diode_orient.npz
    python scripts/diode_orient.py --cv           # grouped by drafter, never by image
    python scripts/diode_orient.py --train        # final model -> models/

``--cv`` and ``--train`` want torch; they run on the GPU when one is there.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASET = ROOT / "data" / "diode_orient.npz"
MODEL = ROOT / "models" / "diode_orient.ts.pt"
REPORT = ROOT / "outputs" / "diode_orient.json"


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

def anode_direction(rotation: float) -> float:
    """CGHD's ``<rotation>`` -> the anode's bearing in image coordinates.

    Straight out of the dataset's own definitions, not guessed:
    ``classes_ports.json`` puts a diode's anode at unit ``x=0`` and its
    cathode at ``x=1``, and ``utils.py``'s ``transform()`` carries a unit
    point into the image by centring it, flipping y, then rotating by
    ``-rotation``. The anode offset ``(-0.5, 0)`` therefore lands along
    ``(-cos r, +sin r)``, whose bearing with y pointing down is ``180 - r``.

    The distinction is invisible on a vertical diode, because ``180 - r == r``
    at 90 and 270 degrees. That is the whole trap: drafter 1's diodes are
    nearly all vertical, so the raw angle looks right there and is inverted
    on every horizontal diode in the rest of the dataset.
    """
    return (180.0 - float(rotation)) % 360.0


def build(out: Path = DATASET, root: Path = ROOT / "data" / "cghd") -> None:
    """Every annotated diode in CGHD, cropped as the pipeline will crop it."""
    from circuit2spice.diode_net import SIZE, crop_symbol, downscale

    xmls = sorted(glob.glob(str(root / "drafter_*" / "annotations" / "*.xml")))
    crops, thetas, drafters, circuits, classes, mirrors = [], [], [], [], [], []
    missing = 0
    t0 = time.time()
    for i, xml in enumerate(xmls):
        try:
            tree = ET.parse(xml).getroot()
        except Exception:
            continue
        boxes = []
        for obj in tree.findall("object"):
            name = (obj.findtext("name") or "")
            if not name.startswith("diode"):
                continue
            bb = obj.find("bndbox")
            rot = bb.findtext("rotation")
            if rot is None:
                continue
            # 277 diodes carry <mirror>vertical</mirror>. It is recorded but
            # not applied: CGHD's own transform() ignores mirror flags, and
            # whether one swaps the ports is a question for the measurement,
            # not for an assumption -- see the per-group score in --cv.
            mir = ",".join(sorted(e.text or "" for e in obj.iter()
                                  if e.tag == "mirror"))
            try:
                boxes.append((
                    float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                    float(bb.findtext("xmax")), float(bb.findtext("ymax")),
                    float(rot), name, mir))
            except (TypeError, ValueError):
                continue
        if not boxes:
            continue

        stem = xml.replace("annotations", "images").rsplit(".", 1)[0]
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            img = cv2.imread(stem + ext)
            if img is not None:
                break
        if img is None:
            missing += 1
            continue
        # Crop at the resolution the pipeline works at, so the strokes the
        # model trains on are as thick as the ones it will be shown.
        img, scale = downscale(img)

        key = Path(xml).stem                         # C<circuit>_D<draw>_P<photo>
        m = re.match(r"(C\d+)", key)
        drafter = re.search(r"drafter_(\d+)", xml.replace("\\", "/")).group(1)
        for x1, y1, x2, y2, rot, name, mir in boxes:
            c = crop_symbol(img, (x1 * scale, y1 * scale, x2 * scale, y2 * scale))
            if c is None:
                continue
            crops.append(c)
            thetas.append(anode_direction(rot))
            drafters.append(int(drafter))
            circuits.append(f"d{drafter}_{m.group(1) if m else key}")
            classes.append(name)
            mirrors.append(mir)
        if i % 250 == 0:
            print(f"  {i}/{len(xmls)} images, {len(crops)} diodes "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, crops=np.array(crops, np.uint8), theta=np.array(thetas, np.float32),
        drafter=np.array(drafters, np.int32), circuit=np.array(circuits),
        cls=np.array(classes), mirror=np.array(mirrors))
    print(f"\nwrote {out}: {len(crops)} diodes over "
          f"{len(set(drafters))} drafters, {missing} images not found")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(x, theta, rng, batch: int, train: bool):
    """Augmented batches. The augmentation *is* the dataset here."""
    from circuit2spice.diode_net import augment

    idx = rng.permutation(len(x)) if train else np.arange(len(x))
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        if train:
            pairs = [augment(x[j], float(theta[j]), rng) for j in b]
            bx = np.stack([p[0] for p in pairs])
            bt = np.array([p[1] for p in pairs], np.float32)
        else:
            bx, bt = x[b], theta[b].astype(np.float32)
        yield bx, bt


def _to_torch(bx, bt, device):
    import torch

    x = torch.from_numpy(bx).to(device).float().div_(255.0)
    x = x.unsqueeze(1).expand(-1, 3, -1, -1)
    r = torch.deg2rad(torch.from_numpy(bt).to(device).float())
    return x, torch.stack([torch.cos(r), torch.sin(r)], 1)


def _fit(xtr, ttr, epochs: int, seed: int = 0, pretrained: bool = True,
         batch: int = 96, lr: float = 3e-4, log=None):
    import torch

    from circuit2spice.diode_net import build_model

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = _device()
    model = build_model(pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, math.ceil(len(xtr) / batch)) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.25)
    model.train()
    done = 0
    for ep in range(epochs):
        tot, n = 0.0, 0
        for bx, bt in _batches(xtr, ttr, rng, batch, True):
            x, t = _to_torch(bx, bt, dev)
            p = model(x)
            p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-6)
            # 1 - cos(error). Scale-free, and exactly the quantity the
            # downstream decision is a sign test on.
            loss = (1.0 - (p * t).sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
            if done <= steps:
                sched.step()
            tot += float(loss) * len(bx)
            n += len(bx)
        if log and (ep % 5 == 0 or ep == epochs - 1):
            log(f"    epoch {ep + 1:>3}/{epochs}  loss {tot / max(n, 1):.4f}")
    model.eval()
    return model


def _predict(model, x, device, batch: int = 256):
    """Angle and agreement for every crop, averaged over four right angles.

    The views are folded back by ``diode_net.combine_tta``, the same call
    inference makes, so what is measured here is what will be run.
    """
    import torch

    from circuit2spice.diode_net import _TTA, combine_tta, rotate_crop

    per_view = []
    for a in _TTA:
        views = np.stack([rotate_crop(c, a) for c in x]) if a else x
        outs = []
        with torch.no_grad():
            for i in range(0, len(views), batch):
                xb = torch.from_numpy(views[i:i + batch]).to(device).float().div_(255.0)
                xb = xb.unsqueeze(1).expand(-1, 3, -1, -1)
                outs.append(model(xb).cpu().numpy())
        per_view.append(np.concatenate(outs))

    ang = np.zeros(len(x))
    agree = np.zeros(len(x))
    for i in range(len(x)):
        got = combine_tta([v[i] for v in per_view])
        if got is not None:
            ang[i], agree[i] = got
    return ang, agree


def _score(ang, agree, theta, min_conf: float):
    """The deployment metric, not the regression one.

    The pipeline knows the diode's axis from its two terminals, so what it
    asks the model is only which way along that axis the anode lies. That is
    a sign test, and it is right exactly when the angular error is under 90
    degrees. ``agree * |cos(error)|`` is the confidence the pipeline will
    actually gate on -- how firmly the model answered, times how well the
    answer lies along the axis the terminals already draw.
    """
    err = np.abs((ang - theta + 180.0) % 360.0 - 180.0)
    along = np.cos(np.radians(err))
    conf = agree * np.abs(along)
    sure = conf >= min_conf
    return {
        "n": int(len(err)),
        "accuracy": float((err < 90.0).mean()),
        "median_error_deg": float(np.median(err)),
        "within_30_deg": float((err < 30.0).mean()),
        "coverage": float(sure.mean()),
        "accuracy_when_sure": float((err[sure] < 90.0).mean()) if sure.any() else 0.0,
    }


def cross_validate(dataset: Path, epochs: int, folds: int, pretrained: bool,
                   min_conf: float, out: Path = REPORT,
                   naive_label: bool = False) -> None:
    """Grouped by drafter, never by image.

    Each circuit is drawn twice and photographed four times, so a random split
    puts eight pictures of the same diode on both sides and measures nothing.
    Holding out whole drafters measures what the model does on a hand it has
    never seen, which is the only number worth quoting.
    """
    d = np.load(dataset, allow_pickle=True)
    x, theta, who = d["crops"], d["theta"], d["drafter"]
    if naive_label:
        # The ablation that names the bug. The stored label is already
        # ``180 - rotation``; undoing it recovers the raw angle, which is what
        # the first attempt trained on. It agrees with the correct label on a
        # vertical diode and is inverted on a horizontal one, so this is not
        # noise -- it is a label error correlated with the very quantity being
        # predicted, which is the worst kind.
        theta = (180.0 - theta) % 360.0
    names = sorted(set(who.tolist()))
    # Round-robin the drafters into folds by size, so no fold is mostly one
    # prolific hand.
    order = sorted(names, key=lambda k: -int((who == k).sum()))
    assign = {k: i % folds for i, k in enumerate(order)}
    print(f"{len(x)} diodes, {len(names)} drafters, {folds} folds, "
          f"{epochs} epochs, pretrained={pretrained}, device={_device()}")

    pred_ang = np.zeros(len(x))
    pred_agree = np.zeros(len(x))
    t0 = time.time()
    for f in range(folds):
        held = np.array([assign[k] == f for k in who.tolist()])
        model = _fit(x[~held], theta[~held], epochs, seed=f, pretrained=pretrained,
                     log=lambda s: print(s, flush=True))
        a, g = _predict(model, x[held], _device())
        pred_ang[held], pred_agree[held] = a, g
        r = _score(a, g, theta[held], min_conf)
        print(f"  fold {f}: n={r['n']:4d}  polarity {r['accuracy']:.3f}  "
              f"median err {r['median_error_deg']:.1f} deg  "
              f"when sure {r['accuracy_when_sure']:.3f} over "
              f"{r['coverage'] * 100:.0f}%   [{time.time() - t0:.0f}s]", flush=True)
        del model

    overall = _score(pred_ang, pred_agree, theta, min_conf)
    per_drafter = {}
    for k in names:
        m = who == k
        per_drafter[str(k)] = _score(pred_ang[m], pred_agree[m], theta[m], min_conf)
    per_class = {}
    for k in sorted(set(d["cls"].tolist())):
        m = d["cls"] == k
        if m.sum() >= 20:
            per_class[k] = _score(pred_ang[m], pred_agree[m], theta[m], min_conf)
    # 277 diodes are annotated <mirror>vertical</mirror> and the labels above
    # ignore it, because CGHD's own transform() does. If that is wrong the
    # group scores near zero rather than near chance, which is unmistakable.
    per_mirror = {}
    if "mirror" in d:
        for k in sorted(set(d["mirror"].tolist())):
            m = d["mirror"] == k
            if m.sum() >= 20:
                per_mirror[k or "(none)"] = _score(pred_ang[m], pred_agree[m],
                                                   theta[m], min_conf)

    print(f"\nout-of-fold over {overall['n']} diodes:")
    print(f"  polarity accuracy      {overall['accuracy']:.3f}")
    print(f"  median angular error   {overall['median_error_deg']:.1f} deg")
    print(f"  when sure              {overall['accuracy_when_sure']:.3f} "
          f"over {overall['coverage'] * 100:.0f}% of diodes")

    out.parent.mkdir(parents=True, exist_ok=True)
    body = dict(overall)
    body.update({"per_drafter": per_drafter, "per_class": per_class,
                 "per_mirror": per_mirror, "naive_label": naive_label,
                 "folds": folds, "epochs": epochs, "pretrained": pretrained,
                 "min_confidence": min_conf, "grouping": "drafter",
                 "minutes": round((time.time() - t0) / 60.0, 1)})
    if naive_label:
        body = {"ablation_naive_label": body}
    prev = {}
    if out.exists():
        try:
            prev = json.loads(out.read_text())
        except Exception:
            prev = {}
    prev.update(body)
    out.write_text(json.dumps(prev, indent=2))
    # The out-of-fold predictions themselves, so a disagreement can be looked
    # at rather than argued about.
    oof = out.with_suffix((".naive.oof.npz" if naive_label else ".oof.npz"))
    np.savez_compressed(oof, angle=pred_ang, agree=pred_agree, theta=theta,
                        drafter=d["drafter"], cls=d["cls"],
                        mirror=d["mirror"] if "mirror" in d else np.array([]))
    print(f"wrote {out} and {oof}")


def train_final(dataset: Path, model_path: Path, epochs: int,
                pretrained: bool = True) -> None:
    """Fit on every drafter and export TorchScript, as the U-Net does.

    TorchScript so that inference needs only ``torch`` -- the project's own
    venv is the CPU build on purpose, and must not have to grow a torchvision
    model definition to load this.
    """
    import torch

    d = np.load(dataset, allow_pickle=True)
    x, theta = d["crops"], d["theta"]
    model = _fit(x, theta, epochs, seed=1234, pretrained=pretrained,
                 log=lambda s: print(s, flush=True))
    a, g = _predict(model, x, _device())
    r = _score(a, g, theta, 0.8)
    print(f"on its own training set: {r['accuracy']:.3f} (not a measurement)")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    cpu = model.to("cpu").eval()
    with torch.no_grad():
        ts = torch.jit.trace(cpu, torch.zeros(1, 3, x.shape[1], x.shape[2]))
    ts = torch.jit.freeze(ts)
    ts.save(str(model_path))
    print(f"wrote {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--scratch", action="store_true",
                    help="no ImageNet weights, for the ablation")
    ap.add_argument("--min-confidence", type=float, default=0.80)
    ap.add_argument("--naive-label", action="store_true",
                    help="train on the raw <rotation>, for the ablation")
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out", type=Path, default=REPORT)
    a = ap.parse_args()
    if a.build:
        build(a.dataset)
    if a.cv:
        cross_validate(a.dataset, a.epochs, a.folds, not a.scratch,
                       a.min_confidence, a.out, a.naive_label)
    if a.train:
        train_final(a.dataset, MODEL, a.epochs, not a.scratch)
    if not (a.build or a.cv or a.train):
        ap.print_help()
