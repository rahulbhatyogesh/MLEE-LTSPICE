# circuit2spice — working notes

Photograph of a hand-drawn circuit → LTspice `.asc` that opens and simulates.
MLEE course project, EEE, BITS Pilani.

```
photo.jpg → YOLOv8 symbol detection → stroke clean-up → wire tracing
          → electrical nodes → SPICE netlist → circuit.asc
```

## Setup

**Use Python 3.12, not 3.14.** Torch/Ultralytics wheels are reliable on 3.12.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[detect,report,data]"
python scripts/fetch_subset.py 1      # drafter 1 + the 12 reference schematics
```

LTspice is optional but it is what *verifies* the output:
`winget install AnalogDevices.LTspice`. Symbol library lands in
`%LOCALAPPDATA%\LTspice\lib\sym`.

Model weights are **not in the repo** (`*.pt` is ignored). Get `best.pt` from the
release/Drive and point `--weights` at it.

## Commands

```bash
pytest tests/ -q                               # 66 tests; LTspice ones skip if absent
circuit2spice run photo.jpg -o out --weights best.pt --verify --simulate
circuit2spice run img.jpg --gt ann.xml -o out  # ground-truth boxes = "perfect detector"

python -m circuit2spice.baseline_svm --root data/cghd
python scripts/benchmark.py                    # end-to-end vs reference netlists
python scripts/benchmark.py --binablate        # pre-processing ablation
python -m circuit2spice.figures                # all figures + results.csv/.tex
python scripts/build_report.py                 # fills report/*.docx
python scripts/build_poster.py                 # fills poster/*.tex
```

Report and poster read their numbers from `outputs/*.json`, so they can never
disagree with what was measured. Re-run the benchmark before rebuilding them.

## Module map (`src/circuit2spice/`)

| File | Role |
|---|---|
| `classes.py` | 59 CGHD labels → 16; LTspice symbol + port count per class |
| `voc2yolo.py` | VOC → YOLO, **drafter-disjoint** splits |
| `detect.py` | YOLO wrapper; `detections_from_voc` = perfect-detector stand-in |
| `wires.py` | binarise, strip ruling, skeletonise, nodes |
| `topology.py` | terminals → nets, junction merge, crossover split |
| `netlist.py` / `asc_writer.py` | graph → SPICE / LTspice |
| `symbols.py` | pin geometry, parsed from LTspice's own `.asy` files |
| `ltspice.py` / `verify.py` | drive `LTspice.exe`; round-trip check |
| `netcompare.py` | graph isomorphism vs reference netlists |
| `figures.py` | every report/poster figure |

## Things that will bite you

**1. LTspice fails silently.** A `.asc` whose wires don't meet still opens fine —
the component just floats. Never trust a schematic you haven't round-tripped:

```python
from circuit2spice.verify import verify_graph
verify_graph(graph, "out.asc").summary()   # 'OK: 10 pins, net partition matches LTspice'
```

Three format facts, each established by probing LTspice, not assumed:
- **Symbol origins must be on the 16-unit grid.** Off-grid symbols get snapped and
  their pins move away from the wires.
- **Pin offsets come from the `.asy` file, not the drawn body.** A resistor's
  second pin is at `(16, 96)`, past the end of the zig-zag.
- **Crossing wires do not connect** — only shared endpoints do.

Connectivity in the output is carried by **net-name `FLAG`s, not routed wires**.
Geometric routing puts an endpoint at every corner, and a corner landing on
another net's segment silently fuses the two. Labels are exact. Don't "improve"
this back into a router without re-running `verify_graph` on everything.

**2. Ruled notebook paper is the dominant failure mode.** Adaptive thresholding
(what the prior work uses) equalises local contrast, which promotes a faint
printed rule line to the status of a pencil stroke. One such line spans the page
and welds every node into one.

Removing it *by connected component* does not work — where a rule line crosses a
wire the two are one component, so you discard the circuit with the ruling.
`remove_thin_strokes()` separates them pixel-by-pixel on the distance transform
(ruling 1–2 px, pencil 3–6 px), so a crossing is **cut** rather than merged.
This is worth 0.104 → 0.333 exact match. Don't regress it.

**3. Split by drafter, never by image.** Each circuit is drawn twice and
photographed four times. A random image-level split puts near-duplicates in train
and test and inflates mAP while measuring nothing.

**4. The trained weights have 21 classes, not 16.** A Colab edit appended to
`NAMES` instead of replacing it, so indices 10–14 are dead duplicates and 15–19
are the ones actually trained. Benign: training was internally consistent and the
model maps ids back to correct *names*, which is what the pipeline keys off.
Don't "fix" it without retraining.

**5. tikzposter sets the title in LR mode.** It will not wrap and a manual `\\`
raises *"Not allowed in LR mode"*. The title has to fit one line (box is 1449 pt).

## Results

96 photographs of CGHD drafter 1 — the only images shipping reference LTspice
schematics (12 circuits × 2 drawings × 4 photos). Scored by graph isomorphism.

| | GT boxes | Full system |
|---|---|---|
| HOG+SVM on isolated crops *(baseline)* | 0.920 | — |
| YOLOv8s detection, mAP@50 *(val drafters)* | — | 0.807 |
| Bill of materials (F1) | 0.966 | 0.965 |
| Local topology (F1) | 0.453 | 0.426 |
| **Exact circuit match** | **0.333** | **0.323** |

Two findings: **finding the parts is solved, wiring them up is not** (0.966 vs
0.333), and **the detector is not the bottleneck** (swapping perfect boxes for the
trained model costs one point).

> Drafter 1 was in the **training** split — it supplies the only reference
> schematics — so the full-system column is optimistic. The GT-boxes column is
> unaffected and is the honest measure of the topology stage. Say so in any
> write-up.

## Open work

- **Component values are not read.** Text boxes are detected and erased but not
  recognised; values default to R=1k/C=1u/L=1m, overridable with `--value R1=4.7k`.
  Biggest remaining win, and every reference paper lists it as future work.
- **Pin order is positional** for 3-terminal devices, so C/B/E may be permuted.
  CGHD ships rotation annotations that would settle this, and would also recover
  diode/source polarity.
- Layout is electrically correct but not draughtsman-quality.
- `drop_off_circuit()` in `wires.py` is **off by default** — it measured as no
  help once thickness filtering was in, and it clips real wires when few symbols
  are detected. Kept for the ablation.

## Conventions

- Report/poster numbers come from `outputs/*.json`. Never hand-type a result.
- Claims about LTspice behaviour get verified against LTspice, not reasoned about.
- Both documents must carry the **same title** (the brief checks this).
- Team names are `[Name 1 (ID)]` placeholders in both documents.
- Poster is laid out A0 but the brief wants an **A3 print** (~2.8× reduction) —
  figure type is oversized deliberately.
