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

Ollama is optional too, and it is what makes label reading fast enough to
demo -- 0.4 s a label against 10 s on the CPU. Without it the pipeline falls
back to the CPU reader in silence:

```bash
winget install Ollama.Ollama
ollama pull qwen3-vl:2b-instruct-q8_0      # 2.7 GB; -instruct, never ":2b"
```

Model weights **are** in the repo, so a clone runs without fetching anything.
Three of them, 125 MB in all — under GitHub's 100 MB per-file limit, which is
why this is possible at all:

- `runs/yolov8s-2/weights/best.pt` — the YOLOv8s detector. Point `--weights`
  at it; the GUI and `scripts/*` find it themselves.
- `models/unet_strokes.ts.pt` — the stroke segmenter, TorchScript so inference
  needs only `torch`. Produced by `colab/train_unet_strokes.ipynb`; the export
  cell runs on CPU, so a spent GPU quota does not block it. Without this file
  `--method unet` raises `StrokeModelMissing`; the thresholding path still
  works. Set `CIRCUIT2SPICE_UNET` to put it elsewhere.
- `models/diode_orient.ts.pt` — the diode orientation head, also TorchScript.
  Produced by `scripts/diode_orient.py --train`. Without it the pipeline
  leaves a diode's terminal order alone, exactly as it did before, so this
  one is optional in the strict sense. `CIRCUIT2SPICE_DIODE` moves it.

**Training the diode model wants CUDA, and the main venv must not have it.**
`.venv` is the CPU build on purpose — gotcha: rebuilding it around CUDA puts
the YOLO detector and the TorchScript U-Net at risk for a stage that already
works. So training uses a *second* venv, which the main one never imports:

```bash
py -3.12 -m venv .venv-gpu
.venv-gpu/Scripts/python -m pip install torch torchvision     --index-url https://download.pytorch.org/whl/cu128
.venv-gpu/Scripts/python -m pip install opencv-python-headless
```

A fold trains in under two minutes on the 4 GB RTX 3050. TorchScript crosses
the version gap fine — traced under 2.11+cu128, loaded under 2.15+cpu.

Segmentation costs ~1.2 s per image on CPU and the model is cached per process.

## Commands

```bash
pytest tests/ -q                               # 457 tests; LTspice ones skip if absent
circuit2spice run photo.jpg -o out --weights best.pt --verify --simulate
circuit2spice run photo.jpg --net 4=Vout --analysis ".tran 1m" --simulate
circuit2spice run photo.jpg --flip D1          # override a polarity by hand
circuit2spice run photo.jpg --as V2=source_current   # correct a misdetection
circuit2spice run photo.jpg --pins M1=N001,N005,0    # state a node order outright
circuit2spice run photo.jpg --supply VCC=5     # drive a rail the drawing never sources
circuit2spice run img.jpg --gt ann.xml -o out  # ground-truth boxes = "perfect detector"

python -m circuit2spice.baseline_svm --root data/cghd
python scripts/benchmark.py                    # end-to-end vs reference netlists
python scripts/benchmark.py --method unet      # ... with learned stroke segmentation
python scripts/benchmark.py --binablate        # pre-processing ablation
python scripts/benchmark.py --repairablate --out outputs/ablation_repair.json
python scripts/eval_values.py                  # component values vs references
python scripts/eval_pins.py                    # transistor pin roles (C/B/E)
python scripts/eval_diodes.py                  # diode polarity, end to end
python scripts/eval_roundtrip.py               # does LTspice accept our .asc?
python scripts/eval_ocr.py --backend vlm       # handwriting, 15 known labels
python scripts/eval_ocr.py --backend ollama    # ... through a local Ollama GPU
CIRCUIT2SPICE_OLLAMA_MODEL=qwen3-vl:4b-instruct python scripts/eval_ocr.py --backend ollama
python scripts/eval_pins.py --positional       # ... without arrowhead reading
python scripts/eval_diodes.py --blind          # ... with the diode model off
python scripts/diode_orient.py --build         # diode crops -> data/
.venv-gpu/Scripts/python scripts/diode_orient.py --cv     # held out by drafter
.venv-gpu/Scripts/python scripts/diode_orient.py --train  # -> models/
python scripts/inspect_circuit.py --circuits C3 C2 C12   # human-checkable pack
circuit2spice gui                              # desktop app: photo -> LTspice
python -m circuit2spice.figures                # all figures + results.csv/.tex
python scripts/build_report.py                 # fills report/*.docx
python scripts/build_poster.py                 # fills poster/*.tex
```

Report and poster read their numbers from `outputs/*.json`, so they can never
disagree with what was measured. Re-run the benchmark before rebuilding them.
The brief's own checks are automated against both outputs: identical title, and
every number on the poster present verbatim in the report.

Three traps found while regenerating them, all of which produce a plausible
document rather than an error:

- **The full-system column must use the same segmentation as the GT-boxes
  column.** `detector.json` holds the older *thresholding* run, so pairing it
  with the learned-segmentation GT numbers showed the detector costing
  0.875 → 0.444 and destroyed the "detection is not the bottleneck" finding.
  Both the table and `fig_result3` now read `benchmark_unet_full.json`; the
  real cost is 3.1 points.
- **Deleting a table by index after inserting one.** `build_report.py` added
  Table 3 and then removed `d.tables[7]` to drop Annex B — but the insertion
  had shifted every later table, so it deleted the *team contributions* table,
  which the course instructions explicitly require. Tables are now matched by
  content.
- **Annex B says "delete this annex before submission"** in the template's own
  guidance, along with the final-checks list. Both are removed automatically.

## Module map (`src/circuit2spice/`)

| File | Role |
|---|---|
| `classes.py` | 59 CGHD labels → 16; LTspice symbol + port count per class |
| `voc2yolo.py` | VOC → YOLO, **drafter-disjoint** splits |
| `detect.py` | YOLO wrapper; `detections_from_voc` = perfect-detector stand-in |
| `wires.py` | binarise, strip ruling, skeletonise, nodes |
| `segment.py` | learned stroke mask (TorchScript U-Net) + its threshold |
| `pins.py` | reads the bar, arrowhead and +/- signs to order C/B/E, D/G/S, +/- |
| `diode_net.py` | regresses a diode's angle, to order anode/cathode |
| `opamp.py` | reads the triangle and the +/- glyphs to order invin/noninvin/out |
| `topology.py` | terminals → nets, junction merge, crossover split |
| `netlist.py` / `asc_writer.py` | graph → SPICE / LTspice |
| `symbols.py` | pin geometry, parsed from LTspice's own `.asy` files |
| `ltspice.py` / `verify.py` | drive `LTspice.exe`; round-trip check |
| `netcompare.py` | graph isomorphism vs reference netlists |
| `values.py` | handwritten labels -> SPICE values; unit picks the class |
| `ocr.py` | reads a label: closed-vocabulary scoring, a local VLM, or Ollama |
| `gui.py` | desktop front end; reports both models, flips a diode, opens LTspice |
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

**2. Ruled notebook paper was the dominant failure mode, and the fix is a
threshold, not a filter.** Adaptive thresholding (what the prior work uses)
equalises local contrast, which promotes a faint printed rule line to the
status of a pencil stroke. One such line spans the page and welds every node
into one.

Removing it *by connected component* does not work — where a rule line crosses
a wire the two are one component, so you discard the circuit with the ruling.
`remove_thin_strokes()` separates them pixel-by-pixel on the distance transform
(ruling 1–2 px, pencil 3–6 px), so a crossing is **cut** rather than merged.
That is worth 0.389 → 0.500 exact match (`outputs/ablation_preproc.json`) and
is still the best the *thresholding* path can do. Don't regress it.

But thickness only separates them when they differ. On C4 the pen and the
printed rules are both 2–3 px, so the filter keeps 91% of the ink, rules
included. The learned segmenter (`method="unet"`) is what closes that case —
**with its threshold at 0.995, not 0.5.** The reason is in the labels: CGHD's
own masks disagree about ruling. Sampling the training drafters, some
annotators erased the rules and some kept them, so the network learns to be
*uncertain* over a rule line — comfortably above 0.5, far below the ~1.0 it
gives a pen stroke. At 0.5 that uncertainty reads as ink and the ruling returns
in full; the exported model reproduces it more crisply than flat-field does.
Raising the cut-off cashes the uncertainty in:

| threshold | 0.5 | 0.9 | 0.99 | **0.995** | 0.998 | 0.9999 |
|---|---|---|---|---|---|---|
| exact match | 0.500 | 0.569 | 0.681 | **0.778** | 0.708 | 0.000 |

The optimum is a plateau (0.708–0.778 over 0.992–0.997), not a spike, and it
collapses past 0.999 as the strokes themselves erode. `thin_q` is skipped
automatically when `method="unet"` — applying it on top only severs thin
pencil.

**Caveat worth stating in the report:** that threshold is tuned on the same 72
images the result is quoted on, because drafter 1 ships the only reference
schematics in CGHD and so is the only end-to-end signal that exists. The
*plateau* is the defensible claim; the exact 0.995 is not.

**3. A symbol's own bounding box erases the branch points on its terminals.**
A box is a rectangle; a terminal sits on its edge. So a wire that *branches at
a terminal* — a transistor emitter feeding both a resistor below it and a rail
to its right — has its branch point **inside** the box, and erasing the box
splits one electrical node into two fragments leaving through adjacent edges.

`reconnect_endpoints()` cannot repair this: the two stubs meet at a right angle
(score 0.36 vs `min_cos` 0.72), so the collinearity evidence is absent, and the
gap crosses an erased box, which it refuses on principle. Widening either rule
is much worse — dropping the box veto alone costs 0.444 → 0.292.

`restore_branch_points()` repairs it instead, using the *surplus* as evidence: a
component has exactly `n_ports` terminals, so if more nodes than that touch its
ring, the extras can only be pieces of one terminal the box cut up. It merges
the closest pair and keeps merging until the count is back to `n_ports`, never
below, so two real terminals can never be shorted.

The split also costs a *second* terminal: `_terminals()` takes the `n_ports`
strongest ring hits by pixel count, so the spurious fourth candidate evicts the
real one. On C2 that was Q1's collector, which left R2 dangling.

**4. A lone ground symbol is not connected to anything.** Drafters mark a
return with a ground symbol and then draw the rail itself without one, and the
reference schematic fuses them — but nothing in the *drawing* joins them, so
wire tracing never can. On C2 drawing 1 that leaves node 0 with a single pin
(V1−) while the real rail carries the other four.

A ground node with one pin is electrically impossible, and that is the only
evidence there is. `fuse_stranded_ground` detects it and merges with the
highest-degree node — the same node the no-ground fallback would have picked.
Ironically it is the *presence* of one ground symbol that breaks it: with none
at all the fallback gets it right.

Both repairs are ablated by `benchmark.py --repairablate` →
`outputs/ablation_repair.json`. They are complementary, not additive: branch
repair alone buys nothing (0.444), ground fusion alone +2 images (0.472),
together +4 (0.500), because on C2 each image was wrong for both reasons at
once.

**5. Exact match cannot see a transistor wired backwards.**
`netlist_to_graph` is undirected and `_signatures` sorts each component's legs,
so a BJT with collector and emitter transposed is *isomorphic* to the correct
circuit. That is the right answer to "is the wiring the same" and the wrong
answer to "will it simulate" — an NPN in reverse-active has a beta of a few
rather than a hundred, so the bias point is wrong while every headline number
stays green. Measured with positional pins: **1/15**.

`pins.py` reads the order out of the drawing in two steps, both forced by what
failed:

- **The base is the lead the bar lies across.** A probe swept along each lead
  meets a long stroke only on the base. The probe has to *slide*: on C2 the bar
  sits hard against the edge of the box, so a single probe through the centre
  misses it entirely.
- **The emitter is the lead carrying the arrowhead**, located as the peak of
  the distance transform. Its *bearing* from the centre is not the lead's
  bearing — the arrow sits near the bar while the terminal is out at the box
  edge, and on CGHD the two differ by up to 70°. What survives the kink is
  which side of the bar's axis it falls on.

**Angles alone cannot do either job.** The three leads commonly leave the box
about 120° apart, and in that configuration "most opposite the other two" and
"most perpendicular to their axis" both score the collector as well as the
base. Two such rules were tried and measured before this one; don't re-derive
them.

The thresholds are calibrated, not guessed: a real arrowhead peaks at 4.8–6.3×
the median stroke thickness, two plain strokes merely crossing reach 2.7×.
When the drawing does not say — no bar that stands out, no arrowhead thicker
than its surroundings — `pins.py` returns the order unchanged rather than
guessing. On CGHD that is 3 of 24 images, and in the other 21 it is never
wrong.

**6. Split by drafter, never by image.** Each circuit is drawn twice and
photographed four times. A random image-level split puts near-duplicates in train
and test and inflates mAP while measuring nothing.

**7. The trained weights have 21 classes, not 16.** A Colab edit appended to
`NAMES` instead of replacing it, so indices 10–14 are dead duplicates and 15–19
are the ones actually trained. Benign: training was internally consistent and the
model maps ids back to correct *names*, which is what the pipeline keys off.
Don't "fix" it without retraining.

**8. tikzposter sets the title in LR mode.** It will not wrap and a manual `\\`
raises *"Not allowed in LR mode"*. The title has to fit one line (box is 1449 pt).

**9. An op-amp's three pins are `invin noninvin out` — not `In+ In- Out`.**
Verified against LTspice rather than reasoned about, because the wrong order
is the one everybody quotes. Flagging a distinct net on each pin of
`OpAmps\opamp` and netlisting it gives

```
X§X1 AAA BBB CCC opamp Aol=100K GBW=10Meg ;§pnba invin)noninvin)out
```

and `opamp.sub` agrees: `.subckt opamp 1 2 3` with `G1 0 3 2 1 {Aol}`, a VCCS
driving node 3 from `V(2) − V(1)`, so node 1 is the *inverting* input. Taking
the other order swaps the inputs, which turns negative feedback into positive:
an inverting amplifier becomes a latch that pins to the rail. It still opens,
still round-trips and still simulates, and `netcompare` cannot see it because
graph isomorphism is undirected.

Three further things that only showed up once an op-amp was actually run:

- **LTspice writes the ref as `X§X1`.** The section sign marks a symbol whose
  pins carry a `SpiceOrder`. `parse_netlist` compared that against our `X1`
  and matched nothing, so *every* op-amp failed the round-trip as "1 component
  absent, 1 unexpected" — the one check meant to catch exactly this class of
  error, defeated by a name. `ltspice._ref()` strips it. The reference C1.net
  carries the same marker on its NE555.
- **A unity-gain buffer loses a pin to `_ring_label_stats`.** It is keyed by
  node label, so an output tied back to the inverting input arrives as *one*
  entry whose centroid is the mean of two contacts on opposite sides of the
  symbol — pointing at neither. The symbol then reported "2/3 terminals" and
  left the output unconnected. `opamp.py` probes the three pin positions
  geometrically instead, so duplicates are allowed.
- **`transistor_fet` has no pin reader at all** and orders D/G/S by skeleton
  pixel count. `pins.py` would plausibly transfer — a MOSFET's gate is the
  lead crossing the gate plate, which is SPICE's middle pin exactly as the
  base is — but it cannot be checked: CGHD annotates **977 FETs and gives
  `<rotation>` for none of them**, and not one of the twelve reference
  netlists contains an `M` element. So the rule is not extended, and
  `build_circuit` warns instead.

## Results

CGHD drafter 1 is the only drafter shipping reference LTspice schematics, so it
is the whole scored set: 12 circuits × 2 drawings × 4 photos = 96 photographs.
Scored by graph isomorphism.

**Every number reported is in-scope: 64 images, 8 circuits.** C1, C6 and C7 are
excluded because they contain parts with no SPICE equivalent — ICs, logic
gates, a motor. "We cannot represent this part" is a different claim from "we
got the wiring wrong", and scoring them together conflates the two. The
unscoped figure is carried in every result JSON under `unscoped`, so the
exclusion is recorded, but it is not what gets quoted.

**C12 is excluded too, by name, and that one is a judgement rather than a
rule** (`EXCLUDED_CIRCUITS` in `benchmark.py`). Its reference uses LTspice's
`sw`, which has 4 nodes against the 2 `classes.py` gives `switch`, so its
reference carries 56 pins where we can emit 52 and isomorphism is impossible
at any image quality. `scope()` cannot derive that on its own: it tests the
element *letter*, and `S` is one we support — it is the arity we do not.

Two things must be said whenever that exclusion is quoted, because the
measurement says them. **C12 would have scored zero on the wiring alone**: its
mean local topology F1 is **0.114** against 0.928 for the other in-scope
circuits, because it is drawn ~3× smaller than the rest (60 px symbols, 33 px
gaps, 2.8 px strokes). And so **excluding it removes the only evidence this
project has about its own resolution floor** — every remaining circuit is
drawn large, so nothing in the scored set now speaks to small drawings at all.
That is the cost of the exclusion and it is not visible in any number.
Deleting `EXCLUDED_CIRCUITS` puts C12 back.

| | GT boxes | Full system |
|---|---|---|
| HOG+SVM on isolated crops *(baseline)* | 0.920 | — |
| YOLOv8s detection, mAP@50 *(val drafters)* | — | 0.807 |
| Bill of materials (F1) | 1.000 | 0.999 |
| Local topology (F1), thresholding | 0.583 | 0.539 |
| Exact circuit match, thresholding | 0.500 | 0.444 |
| Local topology (F1), **learned segmentation** | **0.838** | **0.816** |
| **Exact circuit match, learned segmentation** | **0.778** | **0.750** |
| Transistor pin roles (C/B/E) | **1.000** | `outputs/pins.json` |
| Diode polarity, held out by drafter | **0.994** | 16/16 end to end |
| Op-amp pin roles (invin/noninvin/out) | *unmeasured — see below* | — |

**There is no op-amp number, and there is no way to produce one here.** Only
C1 of the twelve references instantiates an `X` element, and C1 is out of
scope for containing a 555. CGHD annotates 768 op-amps across 33 drafters and
gives `<rotation>` for **none** of them, and `classes_ports.json` has no entry
for `operational_amplifier` at all — so there is no supervision for the pin
roles anywhere in the dataset. `opamp.py` is therefore rule-based like
`pins.py`, not learned like `diode_net.py`: its thresholds are calibrated on
121 CGHD symbols (ink-mass asymmetry, median 0.51 on the correct axis, p10
0.21) and `tests/test_opamp.py` pins the behaviour on synthetic drawings
where the answer is known by construction. Quote it as read-from-the-drawing,
never as an accuracy.

Headline: **56 of 64**. Per circuit, in scope:

| C2 | C3 | C4 | C5 | C8 | C9 | C10 | C11 |
|---|---|---|---|---|---|---|---|
| 7/8 | 8/8 | 8/8 | 8/8 | 4/8 | 8/8 | 6/8 | 7/8 |

C12 is no longer in the scored set (see the scope note above); on the old
72-image set the same 56 images passed, so this is the same result over a
smaller denominator, not 8 new successes. Against thresholding the swap gains 25 images and loses 5; the
losses are C8 (4) and C10 (1), and C8 is the one circuit that is net worse
(gains 2, loses 4) — it is the densest drawing in the set and the higher
cut-off thins its closely-spaced strokes.

Three findings: **finding the parts is solved** (BoM 1.000 throughout);
**the detector is not the bottleneck** — perfect boxes beat the trained model
by 5.6 points under thresholding and only 2.8 under learned segmentation,
against the 50 the topology stage was losing; and **binarisation was the
bottleneck** — replacing four hand-tuned heuristics with one learned function
moved exact match 0.500 → 0.778 (GT boxes) and 0.444 → 0.750 (full system)
with no change to any downstream stage.

Sources: `outputs/benchmark.json` is now the **learned-segmentation** GT-boxes
run, because that is the system as it stands and the report, poster and figures
all read that file. The thresholding run is preserved as
`outputs/benchmark_flatfield.json` and the full-system learned run as
`outputs/benchmark_unet_full.json`. Regenerate both documents after any
benchmark change — they are only as current as the JSON.

> Weights are at `runs/yolov8s-2/weights/best.pt`. Both columns use the same
> settings (`bridge=None`, `drop=none`, `thin_q=50`) so they are comparable —
> `benchmark.py` defaults to `bridge=6, drop=mask` instead, which scores the
> same 0.500 but is not the call `eval_detector.py` makes. Refresh the
> full-system column with `scripts/eval_detector.py`; it now scopes its own
> output and merges into `detector.json` rather than rewriting it, so the
> held-out Colab fields (`map50_test`, `train_minutes`) survive.

Two traps when reading the per-circuit scores:

- **Exact match cannot fail on C3, C4 or C5.** They are small enough that *any*
  rewiring with the right net-size profile is isomorphic to the reference —
  measured accept-rate 1.000 over random rewirings, versus 0.000 for C2, C7, C8
  and C12. C3's 7/8 means "no wire broke", not "the wiring is right". Don't
  quote those circuits as evidence that topology works.
- **C12 cannot pass at any pixel quality**, and it is *in* scope, so its 0/8 is
  a real 8-image hole in the 72. Its reference uses LTspice `sw`,
  which has **4** pins; `classes.py` models `switch` with 2. Reference C12 has
  56 pins, we can only ever emit 52, so the graphs differ in edge count and
  isomorphism is impossible. Feeding the reference netlist back through our own
  port model confirms it: C1–C11 round-trip, C12 does not. Those 8 images are a
  guaranteed zero, and C12 is also drawn ~3× smaller than C3 (60 px symbols,
  33 px gaps, 2.8 px strokes), which is below what `thin_q=50` can survive.

  **Giving `switch` four ports would not rescue C12, and the local topology
  score says so.** Measured per circuit on the GT-boxes run, mean `sig_f1`:

  | C2 | C3 | C4 | C5 | C8 | C9 | C10 | C11 | **C12** |
  |---|---|---|---|---|---|---|---|---|
  | 0.900 | 1.000 | 1.000 | 1.000 | 0.727 | 1.000 | 0.875 | 0.925 | **0.114** |

  0.114 against 0.928 for every other in-scope circuit: the wiring is barely
  recovered at all, quite apart from the pin count. The port model is the
  *stated* reason C12 scores zero and it is real, but it is the second one to
  bite. Fix the switch and C12 is still a circuit whose topology is 0.114, so
  the honest description is a resolution failure wearing a modelling failure
  as a hat — the lever is re-photographing it larger, not editing
  `classes.py`.

> Drafter 1 was in the **training** split — it supplies the only reference
> schematics — so the full-system column is optimistic. The GT-boxes column is
> unaffected and is the honest measure of the topology stage. Say so in any
> write-up.

## Reading the labels

`ocr.py` reads the handwriting as a **closed-vocabulary** problem, not a
transcription one. Free generation asks "what does this say?", whose answer
space is every string in English; with a naming convention we can ask "which
of these does it say?" and score every legal label under the image. The output
then cannot be unparseable -- the worst case is the wrong legal label.

Two constraints come free from things already computed:

- **The detector says the class.** A label sits beside a symbol, so a box next
  to a resistor can only read `R<index>` and only carry ohms.
- **The box shape says the length.** A designator-only label is about as wide
  as it is tall; a designator with a value is three times wider. Estimating
  character count as `(w/h)/0.46` and keeping candidates within +-1.5 of it
  retains every true label on the test drawing while pruning the short decoys
  (`Rout`) that a mean log-probability otherwise prefers.

**Candidates must be written the way the model writes.** TrOCR ends every line
with `" ."`, and a candidate without it is scored in a surface form its
decoder never produces. This is not a detail: on a clean synthetic image of
*hello*, the candidate `hello` scores *below* `zebra`, while `hello .` beats
both. Adding the suffix took the measured score from 2/15 to 5/15.

Measured on one drawing, 15 labels, scoring a label correct only when both its
designator and its value parse to the truth:

| approach | correct |
|---|---|
| easyocr, free | 0 |
| trocr-small, free | 0 |
| trocr-base, free | 0 |
| trocr-base + confusion repair + grammar filter | 5 |
| trocr-base + class-aware rerank | 5 |
| closed-vocabulary scoring, no surface suffix | 2 |
| closed-vocabulary scoring, with suffix | 5 |
| + net labels and the length prior | 7 |
| Qwen2.5-VL-3B, told the convention in its prompt | 4 of the first 4, then killed |
| Qwen2-VL-2B, told the convention | 10 |
| **… + confusion repair over legal labels** | **15** |
| qwen3-vl:2b-instruct-q8_0 on a GPU, 0.4 s/label | 14 |

Every TrOCR arrangement returned the least-wrong *legal* string; the VLM
returns the right one. The last two rows are the same model: all five of the
2B model's residual errors were **one** confusion, the ohm sign, read as `r`,
`n`, `0` or `2` (`10Ω`→`10r`, `6Ω`→`6n`, `R4 1Ω`→`R410`), plus `R5`→`Rs`.
`repair_reading()` substitutes the look-alikes and keeps a result only if it
is a legal label, which is what makes the repair safe rather than hopeful.

It must **rank** the candidates, not take the first that parses: `R3102`
already parses, as designator `R310` with the value 2, so a first-match rule
keeps the wrong reading. `_plausibility` prefers a single-digit index and a
reading that carries a value, which picks `R3` + `10Ω`. There is a test on
exactly that string.

Measured end to end through the pipeline, not replayed: **15/15 at ~10 s per
label** (`outputs/ocr_vlm.json`, from `scripts/eval_ocr.py`), ~150 s for the
drawing's 16 text boxes, inside a 5.4 GB peak working
set for the *whole* run (YOLO, U-Net and reader together). The resident cost
is well under the model's nominal 4.4 GB because safetensors are mmapped and
Windows trims those pages, so working set is the number to quote, not the
parameter count. `VLM_MODEL` therefore defaults to **Qwen2-VL-2B**: it is the
one that was measured, where the 3B was killed before finishing.

The reason is structural. A VLM can be **told** the convention:

> an optional reference like R1, C2, L3, Vin, Vout or Cout, then optionally a
> value like 10k ohms, 100 ohms, 10pF, 4F or 3H

TrOCR has nowhere to put that sentence. Its decoder is a fixed language model,
so the convention can only be applied afterwards, as a filter over an answer
already shaped by English prose.

**Two costs that decide whether it is usable.** Loaded in float32 a 3B model
needs about 14 GB and the first run was killed on a 16 GB machine, so
`VlmReader` defaults to bfloat16 and caps the processor at 200k vision tokens
(`CIRCUIT2SPICE_VLM_DTYPE` overrides the first), because the harness was
upscaling these crops 3x and paying for it. Those two changes took ~120 s per
crop to **~10 s**, which is 2.5 minutes for a drawing: usable in a batch,
still slow in the GUI.

**Speed is the open end, and the GPU is sitting idle.** `torch` in this venv
is the CPU build (`2.15.0+cpu`) and the card is a 4 GB RTX 3050 Laptop, which
could not hold even the 2B model in bfloat16 (4.4 GB) -- so the fix is
quantisation, and rebuilding this venv around CUDA to get it would put the
YOLO detector and the TorchScript U-Net at risk for a stage that already
works. `OllamaReader` takes the other route: Ollama runs the model in **its
own process**, brings its own CUDA, and serves 4-bit builds, so a 4B model is
3.3 GB and fits the card. Our side is an HTTP POST to `127.0.0.1:11434` --
stdlib only, no new dependency -- and `make_reader("auto")` prefers it when
`/api/tags` answers, falling through to `VlmReader` in silence when it does
not, which is the normal case on a machine without Ollama.

`CIRCUIT2SPICE_OLLAMA_MODEL` picks the tag and `CIRCUIT2SPICE_OLLAMA` the
address. Requests go out at temperature 0 -- sampling is precisely how a
reader invents a *plausible* wrong label.

**Tell the reader the unit, never the reference letter.** The scorer uses
the detector's class to *build* its candidate pool; a VLM has to be told, and
what you tell it matters. On a label the drawing writes as `L1 3H`, whose L
and 1 are joined into one stroke, the reader returns `U3N` unprompted and
`U3H` told the unit -- the henries survive even though the index never will.
Naming the letter instead cost **four labels of fifteen**: told "a reference
starts with R", it prepends one that is not there, `10Ω` coming back as
`R10Ω` and `1pF` as `C10pF`. The hint is dropped entirely beside a terminal,
where the label may name a net and the nearest component's class is
unrelated -- with it, `Vout` became `Cout`, its nearest neighbour being a
capacitor.

`repair_reading` then drops a leading letter when one is followed directly by
a digit, so `U3H` yields the value `3H` and the part keeps its positional
name. Ranking makes that safe: `R5` would have to become the unparseable `5`.

**Measured on the same 15 labels** (`scripts/eval_ocr.py --backend ollama`,
one JSON per row in `outputs/`):

| reader | correct | s/label | our RSS |
|---|---|---|---|
| Qwen2-VL-2B, bfloat16, CPU `transformers` | **15/15** | 10.2 | 5.4 GB |
| `qwen3-vl:2b-instruct` (4-bit) | 11/15 | 0.2 | 0.8 GB |
| `qwen3-vl:4b-instruct` (4-bit) | 12/15 | 0.6 | 0.8 GB |
| **`qwen3-vl:2b-instruct-q8_0`** | **14/15** | **0.4** | **0.8 GB** |

**Quantisation decides this, not parameter count.** 8-bit 2B beats 4-bit 4B,
and the labels it recovers are exactly the omega-glyph ones -- `R3 10Ω` read
as `R3 100`, `R4 1Ω` as `R4 1n`. Four bits is what blurs Ω against `0` and
`n`, which is the single distinction this task turns on. The seconds are
*warm*: Ollama keeps the model resident, so the first call after a pull pays
~2 s more. Our own RSS drops to 0.8 GB because the weights are in Ollama's
process, not ours -- end to end the drawing went from 156 s to **14 s** with
the LTspice round-trip still passing (`outputs/demo_ollama/`), which is the
difference between a demo you can run in front of someone and one you cannot.

So `make_reader("auto")` prefers Ollama and the CPU reader is the accuracy
ceiling: `--backend vlm` costs 10 s a label and buys back the one label.

**A fourth trap, which cost an hour and looked like a code regression.**
Ollama decides how much of a model to put on the GPU **once, when it loads**,
from the VRAM free at that moment -- and never revisits it. Train anything on
the same card and the reader that loads during that window stays half on the
CPU for the rest of its life. Measured here on the 4 GB card:

| on the GPU | s/label | the drawing |
|---|---|---|
| 38% (loaded while the diode model was training) | ~10 | ~3 min |
| 58% | 0.57 | 12-15 s |
| fully resident | 0.4 | ~14 s |

Nothing fails. Every character of the output is identical; it is just ten
times slower, so the blame lands on whatever changed most recently. Diagnose
with `curl -s localhost:11434/api/ps` and compare `size_vram` with `size`;
fix by unloading, with the card free, so it reloads properly:

```bash
curl -X POST localhost:11434/api/generate   -d '{"model":"qwen3-vl:2b-instruct-q8_0","keep_alive":0}'
```

`read_structured` now times the reads and `OllamaReader.why_slow` explains a
slow one, which the pipeline raises as a warning. **The trigger is measured
seconds per label, not the offload fraction** -- 58% on the GPU is split and
perfectly fast, so a threshold on the fraction fires when nothing is wrong,
and a warning that cries wolf gets the next real one ignored too.

**Three traps, each of which cost a measurement.**

- **The plain `:2b` and `:4b` tags are *thinking* builds.** Ollama returns
  that reasoning in a separate `thinking` field, so the answer never reaches
  `response`: all 16 boxes read as `""`, `done_reason: length`, the budget
  spent on *"So, let's look at the image. The text is "Cout". Wait,"*. Passing
  `think: false` is not honoured by a thinking-only build. **Always name an
  `-instruct` tag.** `OllamaReader` now raises when `response` is empty while
  `thinking` is not, naming the tag to pull, because a reader returning
  nothing is otherwise indistinguishable from illegible handwriting -- the
  same trap as the Qwen2-VL class mismatch earlier.
- **The prompt's own examples become the fallback answer.** Asked to read a
  crop it could not, the 8-bit reader returned `R1 10k ohms` -- a phrase from
  the prompt, not from the drawing, and an *invented* label is the worst
  possible output because nothing downstream can tell it from a real one.
  Adding "if you cannot read it, reply exactly: UNREADABLE" turned that into
  a blank on both affected boxes. The remaining 1/15 is now a *missing*
  label, not a wrong one, which is the trade this project makes everywhere
  else. It also silenced a spurious duplicate text box the detector finds.
- **Resolution was not the gap.** The local reader is handed these crops 3x
  upscaled, so sending the raw ~148x78 crop looked like the obvious
  asymmetry. Upscaling to 160 px tall measured *worse* -- 10/15 against
  11/15, `10Ω` degrading to `10r2` -- so the crop goes as-is. Don't
  re-derive it.

**`repair_reading` carries two priors beyond the look-alike map**, both forced
by readings measured here:

- **A sub-ohm resistor is not a thing anyone draws.** `R4 1n` parses
  perfectly well as one nanoohm and ties with `R4 1Ω` on every other term,
  so without a magnitude prior the misreading wins on being first. The prior
  is resistance-only on purpose: a drawing that says `C2 10F` means ten
  farads, however absurd that capacitor is.
- **A mangled unit does not always arrive as a look-alike.** The 8-bit reader
  returns `R1 1k-52`, with the glyph expanded into junk rather than
  substituted, so the tail is trimmed and re-ranked. Trimming is safe only
  because `max` returns the *first* maximal candidate and the original is
  first: a trim can only win when the reading as given does not parse at
  all.

**That last sentence was the bug, because it was not true of a bare value.**
`_plausibility` parsed its candidates with no unit, so `1k` -- a string with
no unit written on it -- parsed as *nothing*, scored `-1e9`, and lost to every
rewrite of itself that parsed as something, including the ones built by
deleting its last character and appending the unit. Measured on a drawing that
writes the designator on one line and the value on the next:

| the page says | the reader returned | `repair_reading` stored |
|---|---|---|
| `1k` | `1k` | **`1Ω`** |
| `10k` | `10k` | **`10Ω`** |
| `1m` | `1m` | **`1Ω`** |

Every value on the sheet, off by three orders of magnitude, from a reader that
had read all of them correctly. Nothing downstream could catch it: `1Ω` is a
legal label, so it is the same failure shape as a reversed diode. The fix is
that `_plausibility` now takes the unit the neighbouring class implies -- the
same letter already used to build the prompt and rank the repair -- so the
reading as given parses, ties, and wins on being first.

Two things had to come with it. `complete_unit()` writes the implied symbol
onto the stored text (`1k` -> `1kΩ`), because `values.py` picks a component
for a value *by its unit*, so a bare value was being read correctly and then
dropped on the floor. And `parse_value` no longer lets its `tail` group
swallow an unrecognised trailing letter when it is leaning on an assumed
unit: without that, `10r` parses as "10, assume ohms, ignore the r", which
makes the misreading score as well as `10Ω` and undoes the omega repair the
module was built for. Both are pinned by tests.

The free-running numbers are zero because the decoder is a language model
trained on English prose: it returns `mystiques` for `R1 1k` and `rower` for
`Cout`. trocr-base *is* reading the strokes -- `tooof` for `1000F`, `ton` for
`10ohm`, `ci` for `C1` -- which is why constraining the output space is the
right lever and post-hoc repair is not.

Two corrections worth keeping, because both cost time:
the exhaustive scorer was never "too slow on CPU" -- it was raising
`AttributeError` instantly, because transformers 5 moved
`decoder_start_token_id` to the generation config; and `Vout` cannot be read
from a pool built only from its nearest component's class, because that
component is a capacitor. Net labels belong in every pool.

## Named nets

Every net is `N001`..`N00n` unless the drawing says otherwise, and a
simulation then has to guess which number is the output. A text label beside a
**terminal** symbol names that net instead, so the netlist says `Vout`, the
LTspice `FLAG` says `Vout`, and the `.raw` carries `V(vout)`.

`CircuitGraph.node_name()` is the single choke point every consumer goes
through -- netlist, `.asc`, JSON -- so naming is one dict and cannot get out
of step. `--net 4=Vout` names one by hand, which is what to use until the
recogniser works. Ground is never renamed: SPICE requires node `0`.

Net labels normalise through `_NET_CANON`, because `VOUT` and `Vout` written
on two ends of the same wire would otherwise be two different wires.

**The normalisation was there; the fusion was not, and that was a bug.** Two
nets with the same name is not a clash to be resolved — it is the entire
reason net labels exist. A drafter writes `VCC` at the top of two rails
*instead of* drawing a wire across the page, and nothing on the page joins
them, so wire tracing never can. `name_nets_from_labels` used to skip the
second label as a duplicate name, leaving that rail as `N006` sitting beside
a `VCC` it was meant to be part of: the circuit silently cut in half at the
one point the drawing was being most explicit. It now merges, the same way
`build_circuit` already fuses every ground symbol onto one node. A label
reading `GND` on some other net merges into node 0 for the same reason.

`CircuitGraph.merge_nodes()` retires the dropped id rather than renumbering,
because `n_nodes` counts ids ever issued and renumbering would invalidate
every node number already shown to the user — including the ones `--net` and
`--pins` take.

## Correcting it by hand

Four overrides, and three of them exist because the thing in question is not
learnable from the only dataset there is. They are not a fallback for a
model that needs more training; they are the answer.

| flag | for |
|---|---|
| `--value R1=10k` | a value the reader could not get |
| `--flip D1` | a two-terminal part the wrong way round |
| `--as V2=source_current` | a part whose *class* the detector got wrong |
| `--pins M1=N001,N005,0` | a node order no reader exists for |
| `--pins S1=A,B,CTRL,0` | a switch's control pair, which no drawing shows |
| `--supply VCC=5` | a labelled rail the drawing never sources |

**`--as` exists because CGHD contains 1314 voltage sources and zero current
sources.** The class is in `NAMES` and has no training data behind it at all,
so the detector has never seen a current source, has no competing hypothesis
to offer, and labels one `source_voltage` at 0.91 confidence. Retraining on
this dataset cannot fix that, and no ink statistic can be calibrated for it
either — there are no positive examples to calibrate against. The port count
must match, since a class with more pins would need terminals nothing looked
for; the wiring is untouched, because terminals come from the ink and the ink
does not care what the symbol turned out to be.

**`--as` runs after the designators are taken off the page**, so the ref it
takes is the ref printed in the netlist. Ours are positional until
`apply_names_to_graph` runs and the drawing's own names routinely permute
them — the `V2` you read is not the second source down the page. An override
keyed on an invisible intermediate name would be unusable. Explicit
`--value`s are re-applied afterwards, since the part they name may have
acquired that name a moment earlier.

**`--pins` is the only override that can express a permutation.** `--flip`
reverses, which on a three-pin part swaps the outer two and can never move
the middle one — and a FET with its drain and gate transposed needs exactly
that.

**A switch takes four nodes through `--pins`**, its two terminals and then
the control pair. This is the long-standing `switch` arity problem, solved
where it actually bites rather than by changing `n_ports`: a hand-drawn
switch has two terminals and `_terminals` must not go hunting for two more,
but LTspice's `sw` has four pins and they have to be written. Before this,
`netlist.py` hard-coded them to `0 0` while `asc_writer` left them off the
schematic altogether — the two outputs disagreed about the same circuit, and
in both of them the switch was permanently open. `CircuitGraph.wired_nodes()`
is now the single place that says what a symbol's pins are, and the netlist,
the `.asc` and `verify_graph` all go through it.

**`--as` re-types the detection, not just the component, and then reads the
page again.** Label matching keys off the *detection's* class — `L1` is only
ever offered to an inductor — so an inductor the detector called a resistor
could take neither its designator nor its value off the drawing. Correcting
the class and re-running the name and value passes fixes both, and `--as`
alone is then enough: the two inductors of a boost converter come back as
`L1 100u` and `L2 1m` with nothing else typed.

That needs one more thing, because **`repair_reading` is lossy and
unit-dependent**. Beside a resistor the drawing's `100u` is repaired to
`100Ω` — a 100 microohm resistor is not a thing anyone draws — and the
microhenries cannot be recovered from the repaired string. So
`Detection.text_raw` keeps what the reader actually returned, and
`ocr.rerepair()` redoes the repair against the corrected classes. Nothing is
re-read; the model is not asked again.

## Five silent wrong answers, and what caused each

Every one of these produced a netlist that parsed, opened in LTspice and
passed the round-trip, which is why they survived so long. They were found by
running the pipeline on two drawings and checking the result against the
drawing by hand.

**1. A stroke in the crop ate the value.** The capacitor's own plate landed in
the label crop and the reader prefixed a hyphen: `10u` came back as `-10u`,
`10n` as `-10n`, `1k` as `.1k`. `_VALUE_RE` anchors on a digit, so all three
parse as *nothing*, and a value that parses as nothing is dropped -- leaving
the part on its class default with no trace. Three labels on one drawing, and
`1u` where the page said `10n` puts an oscillator at 45 Hz instead of 4.5 kHz.
`repair_reading` now offers a punctuation-stripped candidate, which is safe
because ranking still has to accept it.

**2. A FET's pins were ordered by ink.** `transistor_fet` had no reader at
all, so D/G/S came out in whatever order `_ring_label_stats` ranked them --
and the *gate* could land anywhere, which meant no single `--flip` could fix
a reversed part. `order_fet_terminals` reads the gate as the lone lead
(`lone_lead_direction`, the same 2-vs-1 structure the op-amp uses, confirmed
by CGHD's own port table: `gate [0, 0.5]`, `drain [0.5, 0]`, `source
[0.5, 1]`). Drain against source is **not** in the ink -- the common MOSFET
symbol is symmetric about its channel, no arrowhead, no bulk -- so that order
is stated as a *convention* in the warnings and a mirrored FET comes out
reversed. The gate being central is what makes `--flip` sufficient.

**3a. An op-amp's sign is not always its own component, and that one is
still open.** `glyph_blobs` finds a `+` or `-` by connected component, which
works when the glyph floats inside the triangle. Drawn cleanly, the input
lead runs through the symbol's edge and straight into the `-`, collinear with
it — one stroke, touching the crop border, dropped. Three op-amps of four on
one rendered schematic decline for that reason.

A fallback was written and **removed after measuring it**. The idea is sound:
both glyphs lie on their lead's own line and only the `+` puts ink anywhere
else, so measure ink off that line just inside the symbol. It passes on
synthetic drawings and returned the *inverted* answer on both real op-amps
that needed it, because the window picks up the triangle's own slanted edge
rather than the glyph. Shipping it would have turned three honest declines
into two confident errors and one correct reading. `pins.sign_by_cross_stroke`
is kept, unused and documented, so nobody re-derives it; narrowing the window
without picking up the edge is the open problem.

**3. Nobody read a source's `+` and `-`.** They are drawn inside the circle,
exactly the glyphs an op-amp marks its inputs with, so `glyph_blobs` finds
both and `order_source_terminals` reads them. Half the sources in any drawing
were backwards before, which is worse than a reversed diode: a supply the
wrong way round inverts every node voltage that depends on it.

**4. A 0.26-confidence ground shorted a rail.** The detector returned both a
`terminal` (0.73) and a `gnd` (0.26) over one VCC symbol, overlapping at
**0.749** against cross-class NMS's 0.75 -- it survived by a thousandth. A
ground symbol is the only detection whose error is not local: every one is
unioned onto a single node, so one false ground does not add a wrong part, it
shorts together any two nets that carry one. The visible symptom was a 12 V
supply with both terminals on node 0. Raising the NMS threshold would tune
the detector on one example; `_shadowed_grounds` refuses to let a
barely-believed ground outvote a confident terminal, which fixes the class.

**5. The net-label whitelist only knew the edges of a drawing.** `_NET_LABEL_RE`
accepted Vin/Vout/VCC/GND and rejected `SW`, `FB`, `GATE`, `OSC`, `CTRL`,
`VLOAD` -- the names a real schematic gives its *internal* nets. Rejecting
them is not neutral, because the point of writing a name on a wire is to join
it to the other wire carrying the same name, so a converter with its feedback
marked `FB` at both ends came back with the comparator input on a one-pin
net. That reads as a wire-tracing failure and is nothing of the kind: the
drawing said so in letters and nothing was listening. Any short identifier
beside a terminal that `parse_label` cannot explain as a part is now a net
name.

## Reading a label is mostly not reading

Four separate bugs, all in `parse_label`/`_MODEL_RE`, all producing a netlist
that simulates. Found by running a 40-part drawing and checking it by hand;
none is visible in the 64-image scored set, because CGHD ships its label text
as annotation and that path never goes near the reader.

**A two-digit designator was read as a part number.** `_MODEL_RE` matches
`R10` — a letter and two digits is exactly the shape of a `2N2222` — and it
was tried *before* the designator branch. So every `R10`..`R99`, `C12`, `R14`
in a drawing became a model name, which is then offered to the nearest
transistor or diode as its value. A MOSFET duly came back as
`M1 N014 N016 0 0 R14`. A bare designator is now decided first.

**The index was being eaten as a value.** The "R110k means R1 10k" split
exists for a *missing* space. The reader returns `R17 Ω` for a label reading
`R17 100k` — index read, value missed, stray ohm sign left behind — and the
split cut `17` into `R1` plus the value `7Ω`. Eight resistors on one drawing
came back holding 0, 1, 2, 3, 4, 5, 6 and 7 ohms, which looks exactly like a
loop counter leaking into the netlist and is nothing of the kind. Two guards:
**a space after the index ends the index**, and the split's value must not be
a bare single digit, because every value on these drawings carries a
multiplier.

That second guard is a judgement, and it is the one case here where something
is lost. `R41Ω` (the omega read into `R4 1Ω`) and `R15Ω` (`R15` with its
value unread) are *the same string*; splitting one means splitting the other.
Preferring the designator costs an order of magnitude on `R410` — recovered
as `R4 10Ω` rather than `R4 1Ω` — and costs **nothing** on the 15-label set,
re-measured at 14/15. It buys eight designators on the 40-part drawing. A
missing value is reported; a wrong one is not.

**A bare unit is an explicable tail.** `R17 Ω` is R17 with its value unread,
so the designator is kept and the part shows up in the "kept the default
value" warning instead of vanishing entirely.

**A rail marker was read as a ground.** A `gnd` and a VCC marker differ by one
stroke and the detector returned `gnd` at 0.58 and 0.46 for the VCC above a
12 V supply and its decoupling capacitor — with no competing `terminal`, so
`_shadowed_grounds` had nothing to compare against. Both parts ended up with
both terminals on node 0: the supply shorted to its own return.
`retag_rail_grounds` uses the one piece of evidence that settles it — nobody
labels a ground symbol VCC — and the graph is then rebuilt from the *cached*
node map, so it costs ring statistics, not segmentation. It fires only on an
explicit contradicting label.

> Widening net names (above) has a sharp edge: `RL` is not a designator this
> module recognises, so `RL1K` — the load resistor's whole label, run together
> by the reader — became a *net* called RL1K. A net name is now letters then
> at most two trailing digits, nothing after: a net does not carry a
> multiplier.

## What was read, and what was guessed

`C2 N003 0 1u` does not record whether that `1u` came off the page or out of
`SPECS`, and the two are the same four characters. On one drawing that said
`10u` and `10n` plainly, both capacitors silently took the `1u` default,
which puts an oscillator at 45 Hz instead of 4.5 kHz — a wrong answer that
looks exactly like a right one.

So `apply_to_graph` records *which* components it actually wrote a reading
onto, and the pipeline reports the rest: `N part(s) kept the default value,
not read from the drawing: C2, C3`. Comparing the value against the default
is not good enough — a 1k resistor the reader got right is indistinguishable
from one that fell back — which is why it is recorded as it happens.

**`--value` names the ref the netlist prints.** It used to be applied three
times over, once inside `build_circuit` where refs are still *positional*.
`--value C2=10u` therefore landed on whichever capacitor was second down the
page, which the drawing then renamed `C1`, destroying its real `100u` while
appearing to do exactly what was asked. It is now applied once, at the end.

**A warning follows a part that gets renamed.** Warnings are raised against
positional refs while the graph is built, and `apply_names_to_graph` then
permutes most of those names onto other parts: `V3 has all terminals on one
node` pointed at a perfectly healthy `V3` while the shorted source was called
`V2`. The warnings are rewritten with the rename map.

**The GUI's Corrections box takes the same thing without the dashes**, and
strips them if they are there, because the obvious thing to do with that box
is paste the command line into it. It used to leave the kind alone on an
unrecognised word, so `--as R10=inductor` quietly became *value*
`R10 = "inductor"`: the part was never re-typed, the netlist read
`R10 N005 N002 inductor`, and nothing said a word. A word it still cannot
place is now reported in the log rather than obeyed as something else, and
a box that parses to *nothing* says so in the status bar -- the worst case
being a box full of text that did nothing, because the netlist then looks as
though the correction was applied and refused rather than never read.

**Whitespace around `=` and `,` is closed up before the line is split.**
`as R10 = inductor` is what a person actually types; splitting it raw gives
three tokens with no pair among them, so the entire box parsed to nothing and
the run went ahead regardless. One consequence worth knowing: a dangling `=`
necessarily absorbs the next word, which is the price of making `R1 = 1k`
work.

On the command line the same spelling with a *space* -- `-- as R10=inductor`
-- cannot work and is not a bug: a bare `--` is POSIX's end-of-options
marker, so everything after it is positional and argparse rejects the lot
with `unrecognized arguments`.

**`--supply` is announced as an invention.** Nothing in the drawing says the
rail voltage, so it is never inferred, and the warning says so in those
words. An undriven rail is not a cosmetic problem: LTspice solves it as
whatever the op-amp input leakage decides.

## The tool

`circuit2spice gui` (or `python -m circuit2spice.gui`) is the demo form: choose
a photograph, watch the stages, and LTspice opens on the result. It finds
weights itself (`models/best.pt`, then `runs/yolov8s*/weights/best.pt`), runs
the pipeline on a worker thread so the window keeps repainting, shows the
netlist and the detection/node overlays, and round-trips the `.asc` through
LTspice before opening it.

**Both models are reported at startup**, the detector and
`models/diode_orient.ts.pt`, and a missing diode model says so in as many
words. It has to: without it the diode keeps positional terminal order, and a
diode the wrong way round opens, verifies and simulates without a murmur. Same
reason the pipeline now appends `read the polarity of N diode(s) from the
drawing` -- a stage that quietly declines is indistinguishable from one that
is not installed.

**The Netlist tab spells a diode as `anode -> cathode`**, because that order
*is* the polarity decision and a bare pair of net names hides it. An op-amp
reads `in- in+ -> out` for the same reason, and needs it more: three bare net
names give no clue which input is inverting, and the consequence of having
them the wrong way round is positive feedback rather than a slightly different
gain.

**`Polarity: [D1 v] Reverse this part`** is the GUI's `--flip`. The model
abstains rather than guesses, so the demo needs the override the command line
has. It reverses the part and rewrites the `.asc`, the `.cir` and the JSON,
then re-verifies -- the photograph is not looked at again, because only the
node order changed and re-detecting would cost half a minute to reach the same
graph.

**The GUI prints which build it is running**, as a path and `gui.py`'s
timestamp, because a window left open across an edit keeps the module Python
already imported and goes on behaving the old way. That is indistinguishable
from the fix not working, and it cost two rounds of exactly that --
diagnosed in the end by noticing the log carried a message string that no
longer existed in the source. Compare the startup line against the file on
disk before concluding anything about a change.

**Tk is not thread-safe, so the worker only ever posts to a queue the UI
drains on a timer.** That sentence was in this file before it was true:
`_work` read `self.v_analysis.get()` and `self.v_verify.get()`, which are Tk
variables, from the worker thread. Under `mainloop()` it races and usually
gets away with it; with no mainloop running it raises
`RuntimeError: main thread is not in main loop` outright, which is how it was
caught -- driving the GUI from a test. Every setting is now read on the main
thread in `run()`/`flip()` and passed to the worker as a plain value. If you
add an option, pass it as an argument; do not reach for the widget.

## Why circuits fail

With thresholding, 36 of 72 passed and the 36 failures were: **21 net splits**
(one net arriving as several fragments), 8 port-count mismatches (C12), 7 lost
terminals. The splits were not small gaps — across all 36 failures only 13
fragment pairs sat within `reconnect_px` and just 3 were T-junctions, so
tuning reconnection was a dead end and endpoint-to-segment repair was not the
missing piece. The breaks were structural, caused by page ruling surviving
binarisation (gotcha 2).

Learned segmentation at the tuned threshold removes most of that bucket:
**56 of 64 pass.** What is left, in order of size:

| bucket | n | |
|---|---|---|
| dense drawing, strokes thinned by the cut-off (C8) | 4 | the one regression |
| remaining net splits (C10 2, C11 1, C2 1) | 4 | |

C12's 8 port-count mismatches used to head this table; they are now outside
the scored set rather than solved. The first row is the cost of the high
threshold and is the obvious thing to attack next — a per-image threshold, or recovering the
`thin_q` idea on the probability map rather than the binary one.

## Open work

- **The U-Net threshold is tuned in-sample.** 0.995 was picked by sweeping
  against drafter 1, which is also what the result is reported on. There is no
  second end-to-end set to tune it against — drafter 1 has the only reference
  schematics — so the honest claim is the plateau (0.708–0.778 across
  0.992–0.997), not the peak. Picking it by IoU against CGHD's masks instead
  would land near 0.5 and undo the gain, because those masks are exactly what
  disagrees about ruling.
- **C8 regressed, 6/8 → 4/8.** It is the densest drawing in the set and the
  high cut-off thins strokes that are already close together. Applying the
  thickness filter to the *probability* map rather than the binary mask was
  tried: `mask = (p > thr) | thick(p > low, q)`. It does what it was built for
  -- C8 recovers to 5/8 -- but every setting costs more elsewhere than it
  gains (best 0.764 against 0.778), and the loss grows monotonically with how
  much of the uncertain band is let back in. The code is in
  `segment.segment_strokes(low_threshold=...)`, **off by default**, with the
  sweep in `outputs/ablation_probthin.json`; the C8 result says the signal is
  real, so a *per-image* choice of cut-off is the live idea, not a global one.
- **PNP is modelled as NPN.** `classes.py` maps every `transistor_bjt` to
  `npn`/`QMOD`, but C8's parts are 2N2907s. `pins.py` already locates the
  arrowhead, and its direction relative to the bar is exactly what separates
  the two, so the information is in hand and unused.
- **A FET's drain and source are still a convention, not a reading.** The
  gate is read (see above); D against S is not, because the common MOSFET
  symbol is symmetric about its channel and carries no arrowhead or bulk
  connection to break the tie. The convention follows CGHD's port table and a
  mirrored FET comes out reversed, which `--flip` settles. There is also
  nothing to measure it against: 977 CGHD FETs carry no `<rotation>` and no
  reference netlist has an `M` element, so a number would mean annotating
  orientation by hand — the diode work could lean on CGHD's existing labels
  and this cannot.
- **`thin_q` was never re-tuned** after the branch/ground repairs landed. On
  the thresholding path `thin_q=70` measured 0.528/0.640 against 50's
  0.500/0.583 — churn rather than a clean win (7 images gained, 5 lost), and
  now largely moot since `thin_q` is skipped under `method="unet"`. The
  default is still 50.
- **Values and names are read from the labels; the recogniser is the gap.**
  `values.py` turns `10KΩ` into `10k` and assigns it to the nearest
  *compatible* component -- the unit decides the class, which is what makes
  assignment survive a dense drawing. Bill-of-values F1 **0.242 -> 0.858**
  (`scripts/eval_values.py`).

  It also reads the **designator**, so the netlist uses the names on the page.
  Three things that took a measurement to get right:

  - A box usually says *both*, `R1 1kΩ`, and parsing only values both
    loses the name and fails on the combined string. `parse_label()` splits it.
  - The designator supplies the unit: `R1 10k` is ohms because of the R, and
    people routinely omit the symbol. A bare `10k` is still rejected.
  - Renaming must be **one batch**. The usual case is a permutation -- our R1
    is their R2 and vice versa, because ours are numbered by page position --
    and renaming one at a time refuses both, since each wanted name is held by
    the other. The netlist then keeps positional names and every value looks
    wrong although every value is right.

  **The recogniser works now; see "Reading the labels" for the numbers**
  (15/15 on the test drawing, Qwen2-VL-2B plus confusion repair). What
  follows is why the obvious choices did not, because all three are still
  installed and all three are still wrong. The detector finds every text box
  and the crops are clean and legible -- `Cout`, `R₁ 1kΩ`, `1000F`, `1pF` --
  so this was never a detection failure. Measured on this project's own
  drawings:

  | backend | what it returns |
  |---|---|
  | `easyocr` | `Cour`, `RITR`, nothing for half the boxes (conf ~0.3) |
  | `trocr-small-handwritten` | `rower`, `mystiques`, `exposure` |
  | `trocr-small` + constrained decoding | `MAGUVIC`, `295452`, `KDKA` |

  `easyocr`'s recogniser is a CRNN trained on printed and scene text.
  TrOCR's failure is more interesting: it is an encoder-decoder whose decoder
  is a language model trained on IAM English prose, so on `R1 1k` it reaches
  for the nearest English *word*. Masking every token containing a character a
  circuit label cannot contain (64002 -> 3566 tokens) removed the words and
  left noise, which says the small model's visual encoder is not reading these
  crops at all -- the prior was not the only problem.

  TrOCR needs `transformers`, plus `sentencepiece` **and** `protobuf` for its
  tokenizer; without both, transformers 5 falls back to a TikToken extractor
  and raises. `_ocr_reader` tries TrOCR first and falls back, and
  `CIRCUIT2SPICE_TROCR` picks the checkpoint.

  **The in-domain option is the one with real data behind it.** CGHD ships
  `<text>` transcriptions: 696 labelled text boxes in drafter 1 alone, so well
  over ten thousand across the dataset, all of them circuit labels rather than
  English prose. That is the same move the U-Net made for binarisation, and
  for the same reason -- the generic model has the wrong prior.

  One bug the crops exposed: detector text boxes clip **horizontally**, so
  `Vin` is boxed from the i onward and no reader could return more than `in`.
  The crop margin now scales with the box's *height* (the character size) and
  is wider than it is tall.

- **Diode polarity is read now, and what was wrong was the label.**
  Four ink statistics were measured over 445 annotated CGHD diodes and the
  best reached 76.2% (the enclosed hole's offset). A ~25k-parameter CNN on
  PCA-canonicalised binary crops then scored **0.589** held out by drafter,
  which is chance. Neither was the real problem.

  **CGHD's `<rotation>` is not the symbol's bearing in the image.** The
  dataset's own `classes_ports.json` puts a diode's anode at unit `x=0` and
  its cathode at `x=1`, and `utils.py`'s `transform()` carries a unit point
  into the image by centring it, **flipping y, and only then** rotating by
  `-rotation`. Composing those, the anode lies along

      180 - rotation         (degrees, image coordinates, y down)

  and not along `rotation`. The two are **equal at 90 and 270 degrees**.
  Drafter 1's diodes are almost all vertical, so drawing the raw angle on six
  of them looks right -- which is exactly how the wrong rule got confirmed by
  eye and then shipped. On a horizontal diode it is inverted, so this is a
  label error *correlated with the quantity being predicted*, which is worse
  than noise because it cannot average out.

  Fixing the label is most of the result. Nothing else differs between these
  two rows:

  | label | polarity | median angular error |
  |---|---|---|
  | raw `<rotation>`, the first attempt's label | 0.695 | 26° |
  | **`180 - rotation`, from the dataset's own port geometry** | **0.994** | **2.7°** |

  **The training loss says it before any metric does.** On the derived label
  the fit settles at 0.017; on the raw one it plateaus at 0.568 and
  stops. A model that cannot fit its own training data is not short of
  capacity when the capacity is a ResNet-18 and the task is one bit -- it is
  being asked to learn a function that is not there, because the same picture
  carries opposite labels depending on which way the diode happens to lie.
  That number was visible in the first attempt too.

  `tests/test_diode.py` pins the convention against a reimplementation of
  CGHD's `transform()`, with horizontal angles in the parametrisation, so it
  cannot regress quietly.

  **The task is posed as angle regression, not as one bit.** That is what
  makes the data enough: rotate the crop by `a` and the target goes to
  `theta - a`; mirror it and the target mirrors. The whole of O(2) is exact,
  closed-form augmentation. It also removes the need to canonicalise -- the
  previous attempt rotated each crop onto the ink's own principal axis, which
  is set by whatever else is in the box, routinely the label written beside
  the part. And it yields a confidence worth having: the pipeline already
  knows the diode's axis from its two terminals, so the predicted angle is an
  *independent* estimate of the same line. `order_diode_terminals` declines
  unless `agreement x |cos|` clears 0.8, where agreement is how
  far four test-time rotations of the crop agree once turned back.

  Measured by 8-fold cross-validation **grouped by drafter, never by
  image** -- 5145 diodes over 32 drafters, `outputs/diode_orient.json`:

  | | |
  |---|---|
  | polarity | **0.994** |
  | median angular error | **2.7°** |
  | within 30° | 0.987 |
  | when sure | 0.998 over 98% |

  One caveat on that last row: the cross-validation has no terminals, so it
  stands in the *true* axis for the one the pipeline would derive from them.
  Coverage in the field is therefore an upper bound -- a diode whose terminals
  came out crooked is declined, and `--blind` against `--learned` in
  `eval_diodes.py` is what measures the whole chain instead of the model alone.

  26 of 32 drafters are perfect and the worst is d14 at
  0.944. By class: zener 1.000,
  plain 0.998, LED
  0.988 -- the LED is hardest because the
  emission arrows beside it are more ink that is not the symbol.

  **The residual is mostly CGHD, not the model.** Every one of the worst
  disagreements is within a few degrees of a full 180 -- the model does not
  drift, it reverses -- and the six largest were looked at: in all six the
  *annotation* points at the bar rather than the triangle. The dataset's
  README says the rotation annotations are work in progress. `fig_diode.png`
  shows them, so the claim can be checked rather than taken.

  **The mirror flag does not swap the ports**, which was a real question: 277
  diodes carry `<mirror>vertical</mirror>` and the labels ignore it, because
  CGHD's own `transform()` does. If that were wrong those 277 would score near
  zero. They score 1.000, against
  0.994 for the rest.

  **End to end**, through the real pipeline against the reference netlists
  (`scripts/eval_diodes.py`, `outputs/diodes.json`): diode polarity
  **16/16**, against
  3/16 with the model off. Only
  C8 and C9 contain diodes *and* are in scope *and* match topologically, so
  that is the whole scorable set -- small, and the only end-to-end diode
  evidence that exists. Note the blind figure is below half: positional order
  is not a coin toss, it is systematically wrong.

  The data is the other half of the story. CGHD has 33 drafters and the
  project had 6 locally, about 500 diodes; `scripts/fetch_subset.py` pulls the
  rest and `--build` turns them into 5145 labelled crops, taken at the
  pipeline's own working resolution (1600 px long side) so the strokes the
  model trains on are as thick as the ones it is shown.

  The model is a ResNet-18 on ImageNet weights with the head **flattened, not
  average-pooled** -- global average pooling is translation-invariant, and
  where the bar sits relative to the triangle is the entire question.

  `--flip D1` still reverses a part by hand, because the model abstains on
  2% of diodes and **source polarity is still unread**.
- **`switch` needs 4 ports** to match LTspice's `sw`; at 2 it makes C12
  unmatchable (see Results). The *writing* half is now done —
  `CircuitGraph.wired_nodes()` emits all four pins and `--pins S1=A,B,CTRL,0`
  sets the control pair — so a switched circuit can be made to simulate. What
  is still open is the *scoring* half: `classes.py` models 2 ports, so a
  reference with 56 pins still meets 52 and C12 stays out of scope. Closing
  that means deciding where the control nodes come from when nobody types
  them, which the drawing does not say.
- Layout is electrically correct but not draughtsman-quality.
- `drop_off_circuit()` in `wires.py` is **off by default** — it measured as no
  help once thickness filtering was in, and it clips real wires when few symbols
  are detected. Kept for the ablation.

## Conventions

- Report/poster numbers come from `outputs/*.json`. Never hand-type a result.
- **Quote in-scope numbers only** (64 images; C1, C6, C7 excluded for having no
  SPICE equivalent, C12 by name for a switch arity we cannot emit). Every
  result JSON also carries `unscoped`, which exists so the exclusion is
  auditable, not so it can be quoted instead. State the C12 exclusion wherever
  the headline is quoted: it is a decision, not a derivation, and it costs the
  project its only measurement of its resolution floor.
- Claims about LTspice behaviour get verified against LTspice, not reasoned about.
- Both documents must carry the **same title** (the brief checks this).
- Team names are `[Name 1 (ID)]` placeholders in both documents.
- Poster is laid out A0 but the brief wants an **A3 print** (~2.8× reduction) —
  figure type is oversized deliberately.
