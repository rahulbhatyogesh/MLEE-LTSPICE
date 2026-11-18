"""Generate every figure and table used by the report and the poster.

Run as ``python -m circuit2spice.figures``. Everything lands in
``outputs/figures`` at 300 dpi, sized to be legible at A3/A0 print.

The report and the poster must show the *same* numbers, so both read from the
JSON files written by ``scripts/benchmark.py`` and ``baseline_svm`` rather than
having any figures hand-edited.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("outputs/figures")

# Validated categorical palette (light surface). Checked with the data-viz
# validator: lightness band, chroma floor, CVD separation and normal-vision
# floor all pass. The sub-3:1 contrast warning is discharged by direct labels.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b6ae"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e7e1",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def _load(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------- 1. pipeline

def fig_method() -> Path:
    """Block diagram of the pipeline (poster 'Methodology', report Figure 1)."""
    fig, ax = plt.subplots(figsize=(12, 3.1))
    ax.set_xlim(0, 100), ax.set_ylim(0, 24)
    ax.axis("off"), ax.grid(False)

    stages = [
        ("Photograph", "hand-drawn\ncircuit", MUTED),
        ("YOLOv8\ndetector", "16 symbol\nclasses", C1),
        ("Stroke\nclean-up", "flat-field +\nthickness filter", C2),
        ("Wire\ntracing", "skeleton ->\nnodes", C2),
        ("Topology", "terminals ->\nnets", C3),
        ("Netlist", "SPICE\nelements", C3),
        ("LTspice\n.asc", "opens &\nsimulates", C4),
    ]
    w, gap = 11.5, 2.3
    x = 1.0
    for title, sub, col in stages:
        ax.add_patch(FancyBboxPatch((x, 7), w, 9, boxstyle="round,pad=0.35",
                                    linewidth=1.8, edgecolor=col,
                                    facecolor=col + "18"))
        ax.text(x + w / 2, 13.2, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(x + w / 2, 9.6, sub, ha="center", va="center",
                fontsize=8.5, color=INK2)
        if x > 1.0:
            ax.add_patch(FancyArrowPatch((x - gap + 0.2, 11.5), (x - 0.25, 11.5),
                                         arrowstyle="-|>", mutation_scale=14,
                                         linewidth=1.5, color=MUTED))
        x += w + gap
    ax.text(50, 3.4,
            "Trained on CGHD (3,269 photographs, 59 labels merged to 16); "
            "verified by round-tripping each .asc through LTspice",
            ha="center", fontsize=9, color=INK2, style="italic")
    return _save(fig, "fig_method.png")


# -------------------------------------------------------------- 2. ablation

ABLATION_LABELS = {
    "adaptive|none|thinNone": "Adaptive\nthreshold\n(prior work)",
    "flatfield|none|thinNone": "Flat-field\nthreshold",
    "flatfield|mask|thin50.0": "+ thickness\nfilter",
}


def fig_ablation() -> Path | None:
    """What each pre-processing step buys, on the 96 annotated photographs."""
    data = _load("outputs/ablation_preproc.json")
    if not data:
        print("  (skipping ablation: outputs/ablation_preproc.json missing)")
        return None

    keys = [k for k in ABLATION_LABELS if k in data]
    if not keys:
        return None
    labels = [ABLATION_LABELS[k] for k in keys]
    exact = [data[k]["exact"] for k in keys]
    sig = [data[k]["sig_f1"] for k in keys]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(keys))
    bw = 0.36
    b1 = ax.bar(x - bw / 2, exact, bw, label="Exact topology match",
                color=C1, edgecolor=SURFACE, linewidth=2)
    b2 = ax.bar(x + bw / 2, sig, bw, label="Local-topology F1",
                color=C2, edgecolor=SURFACE, linewidth=2)
    for bars, vals in ((b1, exact), (b2, sig)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color=INK)
    ax.set_xticks(x), ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Score on 96 photographs")
    ax.set_ylim(0, max(max(exact), max(sig)) * 1.32)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.set_title("Removing printed page ruling is what unlocks topology",
                 fontsize=11.5, fontweight="bold", loc="left", pad=10)
    return _save(fig, "fig_result1.png")


# ------------------------------------------------------------- 3. per-stage

def fig_stage_summary() -> Path | None:
    """Where the pipeline stands stage by stage -- and where it loses."""
    bench = _load("outputs/benchmark.json")
    svm = _load("outputs/baseline_svm.json")
    if not bench:
        return None
    gt = bench.get("gt-boxes") or next(
        (v for v in bench.values() if isinstance(v, dict) and "exact" in v), None)
    if not gt:
        return None

    rows = []
    if svm:
        rows.append(("HOG+SVM on isolated crops\n(baseline, refs [3][4])",
                     svm["accuracy"], C4))
    rows.append(("Bill of materials\n(components found)", gt["bom_f1"], C1))
    rows.append(("Local topology F1\n(per-component nets)", gt["sig_f1"], C2))
    rows.append(("Exact circuit match\n(graph isomorphism)", gt["exact"], C3))

    # Oversized type: the poster is laid out at A0 but the brief asks for an A3
    # print, so every figure is reduced ~2.8x.  Labels sized for on-screen
    # viewing become illegible after that reduction.
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    ax.barh(y, vals, 0.58, color=[r[2] for r in rows],
            edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.018, yi, f"{v:.3f}", va="center", fontsize=16,
                fontweight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="y", visible=False)
    ax.set_title("Finding the parts is solved; wiring them up is not",
                 fontsize=15, fontweight="bold", loc="left", pad=10)
    out = _save(fig, "fig_result2.png")

    # Second panel: does the trained detector cost anything versus perfect boxes?
    det = _load("outputs/detector.json")
    if det:
        # Wide and narrow-barred: at 14pt the value labels of adjacent bars
        # collide if the pairs sit any closer together.
        fig, ax = plt.subplots(figsize=(8.4, 3.6))
        labels = ["Bill of\nmaterials (F1)", "Local\ntopology (F1)",
                  "Exact circuit\nmatch"]
        a = [gt["bom_f1"], gt["sig_f1"], gt["exact"]]
        b = [det["bom_f1"], det["sig_f1"], det["exact"]]
        x = np.arange(3) * 1.35
        bw = 0.42
        r1 = ax.bar(x - bw / 2, a, bw, label="Ground-truth boxes",
                    color=C1, edgecolor=SURFACE, linewidth=2)
        r2 = ax.bar(x + bw / 2, b, bw, label="Trained YOLOv8s detector",
                    color=C2, edgecolor=SURFACE, linewidth=2)
        for bars, vals2 in ((r1, a), (r2, b)):
            for bb, v in zip(bars, vals2):
                ax.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=14,
                        fontweight="bold", color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel("Score on 96 photographs", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 1.25)
        ax.legend(frameon=False, loc="upper right", fontsize=12.5)
        ax.set_title("The detector is not the bottleneck",
                     fontsize=15, fontweight="bold", loc="left", pad=10)
        _save(fig, "fig_result3.png")
    return out


# ------------------------------------------------------- 4. confusion matrix

def fig_confusion() -> Path | None:
    svm = _load("outputs/baseline_svm.json")
    if not svm:
        return None
    labels, cm = svm["labels"], np.array(svm["confusion"], float)
    keep = cm.sum(1) > 0
    labels = [l for l, k in zip(labels, keep) if k]
    cm = cm[keep][:, keep]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    # Sequential = one hue, light -> dark.  Never a rainbow.
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("Predicted"), ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(f"HOG+SVM baseline, {svm['accuracy']*100:.1f}% on crops",
                 fontsize=11, fontweight="bold", loc="left", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.045, label="Fraction of true class")
    return _save(fig, "fig_confusion.png")


# ---------------------------------------------------------- 5. qualitative

def fig_qualitative(stem: str = "C11_D1_P1") -> Path | None:
    """Photo -> detections -> electrical nodes, side by side."""
    import cv2

    from .detect import detections_from_voc, draw
    from .pipeline import _load_and_scale
    from .topology import build_circuit
    from .wires import colorize

    root = Path("data/cghd/drafter_1")
    img_p, xml = root / "images" / f"{stem}.jpg", root / "annotations" / f"{stem}.xml"
    if not img_p.exists():
        print(f"  (skipping qualitative: {img_p} missing)")
        return None

    dets = detections_from_voc(xml)
    img, dets = _load_and_scale(img_p, dets, 1600)
    g, nm = build_circuit(img, dets)

    panels = [(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "(a) photograph"),
              (cv2.cvtColor(draw(img, dets), cv2.COLOR_BGR2RGB), "(b) detected symbols"),
              (cv2.cvtColor(colorize(nm), cv2.COLOR_BGR2RGB), "(c) electrical nodes")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
    for ax, (im, cap) in zip(axes, panels):
        ax.imshow(im), ax.axis("off"), ax.grid(False)
        ax.set_title(cap, fontsize=10.5, color=INK, pad=6)
    nets = ", ".join(
        f"{c.ref}[{' '.join(g.node_name(n) if n >= 0 else 'NC' for n in c.nodes)}]"
        for c in g.components)
    fig.suptitle(f"Extracted netlist:  {nets}", fontsize=9.5, color=INK2, y=0.07)
    return _save(fig, "fig_qualitative.png")


# ------------------------------------------------------------- 6. schematic

def fig_schematic(asc_path: str = "outputs/demo/C11_D1_P1.asc") -> Path | None:
    """Render the generated .asc the way LTspice lays it out."""
    p = Path(asc_path)
    if not p.exists():
        print(f"  (skipping schematic: {p} missing)")
        return None

    wires, symbols, flags = [], [], []
    inst = val = None
    for line in p.read_text(errors="ignore").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "WIRE":
            wires.append([int(v) for v in t[1:5]])
        elif t[0] == "FLAG":
            flags.append((int(t[1]), int(t[2]), t[3]))
        elif t[0] == "SYMBOL":
            symbols.append({"sym": t[1], "x": int(t[2]), "y": int(t[3]),
                            "rot": t[4], "name": "", "value": ""})
        elif t[0] == "SYMATTR" and symbols:
            if t[1] == "InstName":
                symbols[-1]["name"] = t[2]
            elif t[1] == "Value":
                symbols[-1]["value"] = " ".join(t[2:])

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for x1, y1, x2, y2 in wires:
        ax.plot([x1, x2], [-y1, -y2], color=INK, linewidth=1.6, zorder=2)
    for s in symbols:
        ax.add_patch(FancyBboxPatch((s["x"] - 4, -s["y"] - 100), 40, 96,
                                    boxstyle="round,pad=3", linewidth=1.6,
                                    edgecolor=C1, facecolor=C1 + "1a", zorder=3))
        label = s["name"] + (f"\n{s['value']}" if s["value"] else "")
        ax.text(s["x"] + 52, -s["y"] - 52, label, fontsize=8.5,
                va="center", color=INK, zorder=4)
    for x, y, name in flags:
        ax.plot([x], [-y], marker="v" if name == "0" else "o", markersize=7,
                color=C2 if name == "0" else C3, zorder=5)
        ax.text(x + 8, -y - 10, name, fontsize=7.5, color=INK2, zorder=5)
    ax.set_aspect("equal"), ax.axis("off"), ax.grid(False)
    ax.set_title(f"Generated schematic: {p.name}  (net labels carry connectivity)",
                 fontsize=10.5, fontweight="bold", loc="left", pad=10)
    return _save(fig, "fig_schematic.png")


# ------------------------------------------------------------------ tables

def write_tables() -> None:
    bench, svm = _load("outputs/benchmark.json"), _load("outputs/baseline_svm.json")
    abl = _load("outputs/ablation_preproc.json")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [("Method", "Task", "Metric", "Score")]
    if svm:
        rows.append(("HOG+SVM (refs [3][4] reimplemented)", "Classify isolated crops",
                     "Accuracy", f"{svm['accuracy']:.3f}"))
        rows.append(("HOG+SVM (refs [3][4] reimplemented)", "Classify isolated crops",
                     "Macro F1", f"{svm['macro_f1']:.3f}"))
    gt = (bench or {}).get("gt-boxes")
    if gt:
        rows += [
            ("This work (ground-truth boxes)", "Bill of materials", "F1", f"{gt['bom_f1']:.3f}"),
            ("This work (ground-truth boxes)", "Local topology", "F1", f"{gt['sig_f1']:.3f}"),
            ("This work (ground-truth boxes)", "Whole circuit", "Exact match", f"{gt['exact']:.3f}"),
        ]
    if abl:
        for k, lab in ABLATION_LABELS.items():
            if k in abl:
                flat = lab.replace("\n", " ")
                rows.append((f"Ablation: {flat}", "Whole circuit", "Exact match",
                             f"{abl[k]['exact']:.3f}"))

    (OUT / "results.csv").write_text(
        "\n".join(",".join(f'"{c}"' for c in r) for r in rows))
    tex = ["\\begin{tabular}{lllr}", "\\toprule",
           " & ".join(rows[0]) + " \\\\", "\\midrule"]
    tex += [" & ".join(r) + " \\\\" for r in rows[1:]]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "results.tex").write_text("\n".join(tex))
    print(f"  wrote {OUT/'results.csv'} and {OUT/'results.tex'}")


def main() -> None:
    print("generating figures:")
    fig_method()
    fig_ablation()
    fig_stage_summary()
    fig_confusion()
    fig_qualitative()
    fig_schematic()
    write_tables()
    print("done")


if __name__ == "__main__":
    main()
