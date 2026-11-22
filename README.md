# circuit2spice

Turn a photograph of a hand-drawn circuit into an **LTspice schematic that opens
and simulates**.

```
photo.jpg → YOLOv8 symbol detection → stroke clean-up → wire tracing
          → electrical nodes → SPICE netlist → circuit.asc
```

MLEE course project, Department of Electrical and Electronics Engineering,
BITS Pilani.

---

## Why this is not just "classify the symbols"

Every reference this project builds on stops short of a usable circuit:

| Work | Method | Reported | Where it stops |
|---|---|---|---|
| Kumar & Merchant (CS229) | MobileNet transfer, 8 classes | 96% on crops | grid/sliding-window on full schematics "performed poorly"; no netlist |
| Árnadóttir & Yngvason (CS230) | 4-conv CNN, 14 classes | 95.4% crops, 79.4% boxes | HSPICE netlist for **parallel circuits only**; values typed by hand |
| Liu & Xiao | Fourier descriptors + SVM | 94.4% | 5 classes, 6 circuits, no netlist |
| Aksakalli, `circuit_recognizer` | threshold → skeleton → HOG+SVM | 6 classes | emits `.asc`, but hand-engineered throughout |

The CS230 report says plainly: *"Using YOLO was our original intention. However,
this algorithm would require a dataset with labeled bounding boxes, which was
not to be found on the web."*

That dataset now exists — **CGHD** (Bayer et al., DFKI): 3,269 photographs of
hand-drawn circuits, 59 classes, ~248k PASCAL-VOC boxes, CC-BY 4.0. This project
uses it, and goes past detection to the part the references call "a daunting
task": recovering **which wire connects to what**.

## Results

Measured on the 96 photographs of CGHD drafter 1, which are the only images in
the dataset that come with reference LTspice schematics (12 circuits × 2
drawings × 4 photos). Scored by **graph isomorphism** against those references,
not by eye.

| Stage | Metric | GT boxes | Full system |
|---|---|---|---|
| HOG+SVM on isolated crops *(baseline, references reimplemented)* | accuracy | **0.920** | — |
| YOLOv8s symbol detection *(validation drafters)* | mAP@50 | — | **0.807** |
| Bill of materials — are the right components present? | F1 | **0.966** | **0.965** |
| Local topology — is each component wired to the right neighbours? | F1 | **0.453** | **0.426** |
| Whole circuit — is the extracted netlist *the same circuit*? | exact match | **0.333** | **0.323** |

The gap between 0.966 and 0.333 is the real finding: **finding the parts is
solved; wiring them up is not.** Our 0.920 on isolated crops sits right in the
93–96% band the reference papers report — which is the point. That number
measures a much easier problem than the one that matters.

The second finding falls out of the two columns: swapping ground-truth boxes for
the trained detector costs **one point** of exact match. **The detector is not
the bottleneck**, and a bigger or better-tuned one would buy almost nothing.

> Caveat: the 96 annotated photographs belong to drafter 1, who was in the
> *training* split — that drafter supplies the only reference schematics in the
> dataset. The full-system column is therefore optimistic; the GT-boxes column
> is unaffected and is the honest measure of the topology stage.

The detector trained for 48 epochs (early-stopped, best at epoch 36), 172 min on
a Colab T4.

### What moved the needle

| Pre-processing | Exact match | Local-topology F1 |
|---|---|---|
| CLAHE + adaptive threshold *(what the prior work does)* | 0.250 | 0.328 |
| Flat-field division + Otsu | 0.281 | 0.402 |
| **+ stroke-thickness filter** | **0.333** | **0.453** |

Much of CGHD is drawn on **ruled notebook paper**, and adaptive thresholding
equalises local contrast — which is exactly what promotes a faint printed rule
line to the same status as a pencil stroke. Those lines run the width of the
page, so after skeletonisation they weld every node into one and the netlist is
garbage no matter how good the detector is.

Removing them by connected component does not work: where a rule line *crosses*
a wire the two are one component, so any component-level rule throws the circuit
away with the ruling. Thickness separates them pixel-by-pixel instead — printed
ruling is 1–2 px, pencil is 3–6 px — so a crossing is *cut* rather than merged.

## Install

```bash
py -3.12 -m venv .venv                 # 3.12: torch/ultralytics wheels are reliable there
.venv/Scripts/python -m pip install -e ".[detect,report,data]"
```

LTspice is optional but recommended — it is what verifies the generated
schematics. Install from Analog Devices, or `winget install AnalogDevices.LTspice`.

## Use

```bash
# with the trained detector
circuit2spice run photo.jpg -o outputs/demo --weights runs/best.pt --verify

# with ground-truth boxes, to test the topology stage on its own
circuit2spice run data/cghd/drafter_1/images/C11_D1_P1.jpg \
    --gt data/cghd/drafter_1/annotations/C11_D1_P1.xml \
    -o outputs/demo --analysis ".ac dec 50 10 1Meg" --verify --simulate
```

Writes `.asc` (LTspice), `.cir` (SPICE), `.netlist.json`, and overlay PNGs.
Override values with `--value R1=4.7k`.

## Training

`colab/train_yolo_cghd.ipynb` — downloads CGHD, converts VOC→YOLO, trains
YOLOv8s. Open in Colab with a T4; about 75–90 min.

The split is **by drafter, not by image**. Each circuit is drawn twice and
photographed four times, so a random image-level split puts near-duplicates of
the same drawing in train and test and inflates mAP while measuring nothing.

## How correctness is checked

A `.asc` that merely *looks* right is worthless — LTspice fails silently when
wires do not meet, opening the schematic happily with the component floating.
So generated files are handed back to LTspice:

```python
from circuit2spice.verify import verify_graph
verify_graph(graph, "out.asc").summary()   # 'OK: 10 pins, net partition matches LTspice'
```

`LTspice.exe -netlist` is asked what circuit it sees, and that is compared
against the circuit we meant to describe — on the *partition of pins into nets*,
since LTspice numbers nets its own way.

Three facts about the format, each established by netlisting probe schematics
rather than assumed:

1. **Symbol origins must sit on the 16-unit grid.** Off-grid symbols are
   silently snapped and their pins move away from the wires.
2. **Pin offsets come from the symbol file, not the drawn body.** A resistor's
   second pin is at `(16, 96)`, past the end of the zig-zag.
3. **Crossing wires do not connect** — only shared endpoints do.

Connectivity in the output is carried by **net-name labels** rather than routed
wires. Geometric routing puts a wire endpoint at every corner, and if a corner
lands on another net's segment LTspice fuses them; labels are exact.

```bash
pytest tests/ -q        # 66 tests; LTspice round-trips skip if it is not installed
```

## Reproducing the numbers

```bash
python scripts/fetch_subset.py 1                    # drafter 1 + reference schematics
python -m circuit2spice.baseline_svm --root data/cghd
python scripts/benchmark.py                          # end-to-end vs reference netlists
python scripts/benchmark.py --binablate              # the pre-processing ablation
python -m circuit2spice.figures                      # every figure and table
```

## Layout

```
src/circuit2spice/
  classes.py      59 CGHD labels → 16; LTspice symbol map
  voc2yolo.py     VOC → YOLO, drafter-disjoint splits
  detect.py       YOLO wrapper; ground-truth boxes as a "perfect detector"
  wires.py        binarise, strip ruling, skeletonise, nodes
  topology.py     terminals → nets, junction merge, crossover split
  netlist.py      graph → SPICE
  asc_writer.py   graph → LTspice .asc
  symbols.py      pin geometry, read from LTspice's own .asy files
  ltspice.py      drive LTspice.exe (-netlist, -b -Run)
  verify.py       round-trip check against LTspice
  netcompare.py   graph isomorphism vs reference netlists
  evaluate.py     structural topology metrics
  figures.py      all report/poster figures
```

## Limitations

- **Component values are not read.** Text boxes are detected and erased, but the
  handwriting inside them is not recognised; values default to R=1k, C=1u, L=1m
  and are editable with `--value`. Every reference paper lists this as future
  work too.
- **Source polarity is not recovered.** An axis-aligned box does not say which
  way a voltage source points, and nothing reads it yet; `--flip V1` reverses a
  part's terminals by hand. Transistor pin order and diode polarity *are* read
  from the drawing now -- the first from the bar and the arrowhead
  (`pins.py`), the second by a small model trained on CGHD's rotation
  annotations (`diode_net.py`) -- but both abstain rather than guess when the
  ink does not say, so the override still matters.
- **Layout is functional, not draughtsman-quality.** Components keep their
  relative positions from the photograph.
- ICs, logic gates, transformers and similar (~4% of boxes) are detected as
  `other_component` so their ink can be erased, but they are not netlisted.

## Data

CGHD, CC-BY 4.0 — Bayer, J., et al. *A Public Ground-Truth Dataset for
Handwritten Circuit Diagram Images.* DOI `10.5281/zenodo.15333233`;
HuggingFace mirror `lowercaseonly/cghd`.
