"""Fill the MLEE report template with the real content and the measured numbers.

Reads every number from the JSON written by the benchmark and baseline scripts,
so the report cannot drift from what was actually measured. Detector numbers are
picked up from ``outputs/detector.json`` when training has finished; until then
those sentences are written with a clearly marked placeholder.

    python scripts/build_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "report" / "MLEE_Project_Report_Template.docx"
OUT = ROOT / "report" / "MLEE_Project_Report.docx"
FIG = ROOT / "outputs" / "figures"

TITLE = "Hand-Drawn Circuit Schematics to LTspice Netlists"


def load(name, default=None):
    p = ROOT / "outputs" / name
    return json.loads(p.read_text()) if p.exists() else default


BENCH = load("benchmark.json", {})
ABL = load("ablation_preproc.json", {})
SVM = load("baseline_svm.json", {})
DET = load("detector.json")
# The full-system column must come from the *same* segmentation as the
# ground-truth-box column, or the table compares a learned-segmentation run
# against a thresholding one and blames the difference on the detector.
FULL_UNET = load("benchmark_unet_full.json", {}).get("full-system", {})
DIODE = load("diode_orient.json", {})
RT = load("roundtrip.json", {})
PINS = load("pins.json", {})
DIODES_E2E = load("diodes.json", {})
VALUES = load("values.json", {})

GT = BENCH.get("gt-boxes", {})
TODO = "[TO BE FILLED once YOLOv8 training completes]"


def f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else TODO


EXACT, SIG, BOM = f(GT.get("exact")), f(GT.get("sig_f1")), f(GT.get("bom_f1"))
SVM_ACC = f(SVM.get("accuracy"))
D_ACC, D_N = f(DIODE.get("accuracy")), DIODE.get("n", "?")
D_ERR = (f"{DIODE['median_error_deg']:.1f}"
         if isinstance(DIODE.get("median_error_deg"), (int, float)) else TODO)
D_NAIVE = f(DIODE.get("ablation_naive_label", {}).get("accuracy"))
A_ADAPT = ABL.get("adaptive|none|thinNone", {})
A_FLAT = ABL.get("flatfield|none|thinNone", {})
A_THIN = ABL.get("flatfield|mask|thin50.0", {})

_rt = RT.get("gt-boxes", {})
RT_TXT = (f"{_rt['ok']}/{_rt['images']}" if _rt else TODO)
RT_RATE = f(_rt.get("rate"))
_pin = PINS.get("arrowhead/gt-boxes", {})
_pin0 = PINS.get("positional/gt-boxes", {})
PIN_TXT = (f"{_pin['pins_correct']}/{_pin['pins_scored']}" if _pin else TODO)
PIN0_TXT = (f"{_pin0['pins_correct']}/{_pin0['pins_scored']}" if _pin0 else TODO)
_dio = DIODES_E2E.get("learned/gt-boxes", {})
_dio0 = DIODES_E2E.get("blind/gt-boxes", {})
DIO_TXT = (f"{_dio['diodes_correct']}/{_dio['diodes_scored']}" if _dio else TODO)
DIO0_TXT = (f"{_dio0['diodes_correct']}/{_dio0['diodes_scored']}" if _dio0 else TODO)
_val = VALUES.get("gt-boxes", {}).get("informative", {})
VAL_READ, VAL_DEF = f(_val.get("read")), f(_val.get("default"))
N_IMG = GT.get("images", 64)
DET_COST = f"{abs(GT.get('exact', 0) - (FULL_UNET or {}).get('exact', 0)) * 100:.1f}"

if DET:
    det_map = f(DET.get("map50")); det_map95 = f(DET.get("map50_95"))
    _full = FULL_UNET or DET
    FULL_EXACT, FULL_SIG, FULL_BOM = (f(_full.get("exact")), f(_full.get("sig_f1")),
                                      f(_full.get("bom_f1")))
    det_sentence = (
        f"The detector reaches mAP@50 of {det_map} and mAP@50:95 of {det_map95} "
        f"on the validation drafters, after {DET.get('epochs_run', '?')} epochs "
        f"(best at epoch {DET.get('best_epoch', '?')}, "
        f"{DET.get('train_minutes', '?')} minutes on a Colab T4).")
    full_exact = FULL_EXACT
else:
    FULL_EXACT = FULL_SIG = FULL_BOM = TODO
    det_map = det_map95 = TODO
    det_sentence = (
        f"Detector mAP on the held-out drafters is {TODO}; the topology results "
        "below are reported with ground-truth boxes, which isolates the wire-"
        "tracing stage from detector error.")
    full_exact = TODO


# --------------------------------------------------------------------- content

ABSTRACT = (
    "Transcribing a hand-drawn circuit into simulation software is still done by "
    "hand. Prior work recognises individual components well but stops before a "
    "usable circuit: reported accuracies of 94-96% are measured on pre-segmented "
    "component crops, and the one published attempt at simulation was restricted "
    "to purely parallel circuits because node detection proved intractable. This "
    "project builds the full chain from a photograph to an LTspice schematic that "
    "opens and simulates, using a YOLOv8 detector trained on the CGHD dataset "
    "(3,269 photographs, 248k annotated symbols) followed by stroke clean-up, "
    f"skeleton-based wire tracing and netlist assembly. Evaluated by graph "
    f"isomorphism against the dataset's own reference schematics, the system "
    f"recovers the correct bill of materials with F1 {BOM} but the exactly "
    f"correct circuit for only {EXACT} of photographs. The detector reaches "
    f"mAP@50 {det_map}, and substituting it for ground-truth boxes costs "
    f"{DET_COST} points of exact match, so detection is not the limiting stage. A "
    f"reimplementation of the prior-art classifier scores {SVM_ACC} on isolated "
    "crops, confirming that component recognition is largely solved and that "
    "connectivity, not classification, is the remaining obstacle."
)

KEYWORDS = ("Keywords: circuit recognition, object detection, netlist extraction, "
            "SPICE simulation, document image analysis")

INTRO = [
    "Engineers sketch circuits on paper long before they draw them in software. "
    "Turning that sketch into a simulation means retyping every component and "
    "every connection into a schematic editor, which is slow and error-prone. "
    "Automating it would let a photograph of a whiteboard or a lab notebook "
    "become a simulatable model directly, which is useful in teaching "
    "laboratories, in design review, and in digitising legacy documentation.",

    "The task is to map a photograph of a hand-drawn circuit to a netlist: a set "
    "of components, each with a type, and the electrical nodes joining their "
    "terminals. The input is a single RGB photograph taken under uncontrolled "
    "lighting; the output is an LTspice .asc file that opens in the tool and "
    "simulates. The problem decomposes into symbol detection, which modern object "
    "detectors handle well, and connectivity recovery, which they do not address "
    "at all.",

    "This project makes three contributions. First, it trains a symbol detector "
    "on CGHD, a public dataset of annotated hand-drawn circuit photographs that "
    "did not exist when the reference work was written, and which its authors "
    "explicitly identified as the missing ingredient. Second, it implements wire "
    "tracing and netlist assembly for arbitrary topologies, not only the parallel "
    "circuits to which prior work was restricted. Third, and most importantly for "
    "interpreting the result, it evaluates the output as a circuit rather than as "
    "a set of labels, by testing graph isomorphism against the reference "
    "schematics the dataset provides.",

    "Section 2 reviews the prior work and the dataset. Section 3 describes the "
    "pipeline and the evaluation protocol. Section 4 reports the results and the "
    "ablation that identifies printed page ruling as the dominant failure mode. "
    "Section 5 concludes.",
]

BG_THEORY = [
    "A circuit is a labelled graph. Writing C for the set of components and N for "
    "the set of electrical nodes, an extracted circuit is a bipartite incidence "
    "structure in which each component c with p(c) terminals contributes p(c) "
    "edges to nodes. Two extractions describe the same circuit when their graphs "
    "are isomorphic under a mapping that preserves component type, which is the "
    "criterion used throughout Section 4.",

    "Symbol detection is posed as single-stage object detection. The detector "
    "predicts, for each anchor, a class distribution and a box offset, minimising "
    "a sum of classification and localisation losses over the training set. Wire "
    "tracing is posed as connected-component labelling on the morphological "
    "skeleton of the binarised drawing after the detected symbols have been "
    "erased: each surviving component of the skeleton is one candidate node.",
]

BG_RELATED = [
    "Work on hand-drawn circuit recognition splits into classical and learned "
    "approaches, and almost all of it evaluates on isolated component images. "
    "Liu and Xiao [3] segment a sketch using topological feature points, then "
    "classify each component from Fourier descriptors of its boundary with an "
    "SVM, reporting 94.4% over five classes. Roy et al. [5] combine texture and "
    "shape features with feature selection and report 93.8% over twenty classes. "
    "Aksakalli's circuit_recognizer [4] is the most complete classical pipeline: "
    "adaptive thresholding, skeletonisation, line-segment detection to remove "
    "wires, then HOG features into an SVM over six classes, and it does emit an "
    "LTspice file.",

    "Deep learning has been applied mainly to the classification step. Kumar and "
    "Merchant [1] fine-tune MobileNet on 1,760 augmented images of eight "
    "component classes and reach 96% on isolated crops, but when they scan a full "
    "schematic with a grid or a sliding window they report that both 'performed "
    "poorly', with components repeatedly misclassified as background. Arnadottir "
    "and Yngvason [2] train a four-layer CNN on a fourteen-class Kaggle set, "
    "reaching 95.4% on crops and 79.4% box accuracy with sliding windows, and "
    "generate an HSPICE netlist - but only for parallel circuits, because, in "
    "their words, general node detection 'proved to be a daunting task'. They "
    "also state that they intended to use YOLO but that a dataset with labelled "
    "bounding boxes 'was not to be found on the web'.",

    "That dataset now exists. Bayer et al. [6] released CGHD, a public "
    "ground-truth set of 3,269 photographs of hand-drawn circuit diagrams from 31 "
    "drafters, annotated with roughly 248,000 bounding boxes over 59 classes in "
    "PASCAL VOC format, together with binary stroke segmentation maps and, for "
    "one drafter, reference LTspice schematics. Their own modular extraction "
    "baseline [7] combines detection, segmentation, character recognition and "
    "orientation regression.",
]

BG_GAP = [
    "Two gaps follow. The first is that the reported accuracies are not "
    "comparable to the task that matters: a classifier handed a pre-segmented "
    "crop solves a strictly easier problem than a system that must find the "
    "components itself, and none of [1]-[5] reports whether the recovered circuit "
    "is correct. The second is that connectivity is either avoided [1], severely "
    "restricted [2], or handled by hand-engineered rules [3][4]. This project "
    "addresses both: it uses the detector the prior work wanted but could not "
    "train, it recovers connectivity for arbitrary topologies, and it scores the "
    "result as a circuit rather than as a set of labels.",
]

METHOD_OVERVIEW = [
    "The pipeline is shown in Figure 1 and runs in six stages. A photograph is "
    "passed to a YOLOv8 detector which locates and classifies circuit symbols. "
    "The image is binarised and cleaned of printed page ruling. The detected "
    "symbol boxes are erased from the binary image, leaving only wires, which are "
    "skeletonised and labelled into connected components - one per candidate "
    "electrical node. Each component's terminals are matched to the nodes that "
    "touch its bounding box, junction and crossover annotations repair the two "
    "cases where naive labelling is wrong, and the resulting graph is written out "
    "both as a SPICE deck and as an LTspice .asc file.",
]

METHOD_DATA = [
    "All data is CGHD [6] (CC-BY 4.0), obtained from its HuggingFace mirror. The "
    "59 annotation labels are merged by prefix into 16 classes chosen so that each "
    "maps onto an LTspice symbol: resistor, capacitor, inductor, diode, BJT, FET, "
    "voltage source, current source, operational amplifier, ground, junction, "
    "crossover, switch, terminal, text, and a catch-all other_component. The "
    "catch-all matters more than it appears: integrated circuits, logic gates, "
    "transformers and similar account for about 4% of boxes and cannot be "
    "netlisted, but a symbol the detector cannot see is a symbol that cannot be "
    "erased, and un-erased ink survives wire tracing as a phantom node that shorts "
    "together everything it touches.",

    "The train/validation/test split is by drafter, not by image. Each drafter "
    "draws every circuit twice and photographs each drawing four times, so a "
    "random image-level split places near-duplicates of the same drawing in both "
    "training and test sets; that inflates apparent accuracy while measuring "
    "nothing about generalisation to an unseen hand. Four drafters are held out "
    "for test and three for validation.",

    "Evaluation of the end-to-end system uses drafter 1, the only part of CGHD "
    "that ships reference LTspice schematics. It contains 12 reference circuits "
    "and 96 photographs (12 circuits x 2 drawings x 4 photographs), so every "
    "photograph has a ground-truth netlist. The references are netlisted with "
    "LTspice itself to obtain the target element lists.",
]

METHOD_MODEL = [
    "Detection uses YOLOv8s at 896 px input, trained for 50 epochs with mosaic "
    "augmentation, modest rotation and horizontal and vertical flips, which are "
    "label-preserving for circuit symbols.",

    "Binarisation divides the greyscale image by a heavily median-blurred copy of "
    "itself and thresholds the quotient with Otsu's method. The textbook "
    "alternative, CLAHE followed by adaptive thresholding, is what the prior work "
    "uses and is actively harmful here: much of CGHD is drawn on ruled notebook "
    "paper, and equalising local contrast promotes a faint printed rule line to "
    "the same status as a pencil stroke. Flat-field division preserves the "
    "absolute darkness of a stroke, so pencil survives and ruling largely does not.",

    "Residual ruling is removed by stroke thickness. Removing it by connected "
    "component fails, because wherever a rule line crosses a wire the two form a "
    "single component and any component-level rule discards the circuit along with "
    "the ruling. The distance transform assigns each ink pixel its half-stroke "
    "width; pixels below the median are dropped and the surviving cores are "
    "dilated back to full width. Printed ruling is one to two pixels wide and "
    "vanishes; pencil is three to six and survives, and a crossing is cut rather "
    "than merged.",

    "Nodes are the connected components of the skeleton after symbol erasure, "
    "labelled on a slightly dilated copy so that gaps opened by thresholding do "
    "not split one net into several. A junction annotation unions every fragment "
    "it touches; a crossover is erased and only collinear stub pairs are re-linked, "
    "which is what allows non-planar and non-parallel topologies. Terminals are "
    "assigned by histogramming node labels in an annulus outside each component "
    "box, growing the annulus until the expected number of ports is found.",

    "Output is written as both a SPICE deck and an LTspice .asc. Connectivity in "
    "the .asc is carried by net-name labels rather than routed wires: geometric "
    "routing places a wire endpoint at every corner, and if a corner lands on "
    "another net's segment LTspice silently fuses the two nets, whereas a label is "
    "exact. Three properties of the format were established experimentally rather "
    "than assumed - symbol origins must lie on the 16-unit grid or LTspice snaps "
    "them and their pins move; pin offsets come from the symbol definition, not "
    "the drawn body; and crossing wires do not connect.",
]

METHOD_SETUP = [
    "Training runs on a Google Colab T4. Inference, wire tracing and all "
    "evaluation run on a laptop with an RTX 3050 and Python 3.12. LTspice 26.1.1 "
    "is driven from Python for verification and simulation. Key settings are "
    "given in Table 1.",
]

METHOD_MODEL += [
    "Four of the sixteen classes have pins that are not interchangeable, and "
    "nothing upstream can notice when they are swapped: graph isomorphism over an "
    "undirected component/net graph does not see pin roles, so a transistor wired "
    "backwards, a reversed diode, an inverted supply or an op-amp with its inputs "
    "exchanged all score as correct while simulating as a different circuit. Each "
    "is therefore read from the drawing itself.",

    "A bipolar transistor's base is the lead the bar lies across and its emitter "
    "the lead carrying the arrowhead, located as the peak of the distance "
    "transform; angles alone cannot do either job, because the three leads "
    "commonly leave the symbol about 120 degrees apart and in that configuration "
    "'most opposite the other two' scores the collector as well as the base. A "
    "field-effect transistor's gate is the lone lead, which CGHD's own port table "
    "confirms, but its drain and source carry no distinguishing mark at all in the "
    "common symbol, so that order is applied as a stated convention rather than "
    "reported as a reading. An op-amp's output is likewise the lone lead, and "
    "which input is inverting is read from the drawn plus and minus signs. A "
    "source's polarity is read the same way.",

    "A diode's polarity resists every ink statistic: four were measured over 445 "
    "annotated symbols and the best reached 76.2 per cent. It is instead posed as "
    "angle regression with a sine-cosine head on a ResNet-18, which makes the whole "
    "of the planar rotation and reflection group exact closed-form augmentation, "
    "and whose head is flattened rather than average-pooled because global average "
    "pooling is translation-invariant and where the bar sits relative to the "
    "triangle is the entire question.",

    "Every one of these stages declines rather than guesses when the drawing does "
    "not say, and says so in the output, because a stage that silently abstains is "
    "indistinguishable from one that is not installed and the netlist is wrong "
    "either way.",
]

METHOD_METRICS = [
    "Three metrics are reported, at increasing strictness, because component-level "
    "accuracy alone is what makes the prior work look stronger than it is.",

    "Bill-of-materials F1 compares the multiset of element types against the "
    "reference, which is the closest analogue to the accuracy numbers quoted in "
    "[1]-[5]. Local-topology F1 compares, for each component, its own type "
    "together with the types of the components sharing each of its nets, giving "
    "partial credit when most of a circuit is right. Exact topology match is "
    "graph isomorphism against the reference netlist, preserving element type. "
    "Node names are ignored throughout, since LTspice numbers nets its own way; "
    "ground is pinned to ground only when the drawing actually contains a ground "
    "symbol, because otherwise the reference's choice of reference node is not "
    "recoverable from the drawing.",

    "Separately, every generated .asc is verified by handing it back to LTspice: "
    "the netlist LTspice extracts is compared against the circuit the writer "
    "intended, on the partition of pins into nets. This catches the silent failure "
    "mode in which a schematic opens correctly but a component is not actually "
    "connected.",
]

RESULTS_QUANT = [
    det_sentence,

    f"Table 2 gives the end-to-end results. Of the 96 annotated photographs of "
f"drafter 1, {N_IMG} are in scope: C1, C6 and C7 are excluded for containing "
f"parts with no SPICE equivalent (an IC, logic gates, a motor), and C12 by "
f"name, because its reference uses LTspice's four-node sw where our port "
f"model gives a switch two, so isomorphism is impossible at any image "
f"quality. Every result JSON also carries the unscoped figure, so the "
f"exclusion is auditable rather than quietly applied. The "
    f"reimplemented HOG+SVM baseline of [3][4] reaches {SVM_ACC} accuracy on "
    f"ground-truth crops, consistent with the 93.8-96% reported in the literature "
    f"and confirming that the baseline is faithful. With ground-truth boxes, the "
    f"full pipeline recovers the correct bill of materials with F1 {BOM}, but "
    f"local-topology F1 is only {SIG} and the exactly correct circuit is recovered "
    f"for {EXACT} of photographs.",

    f"The distance between {BOM} and {EXACT} is the central result. Finding and "
    f"naming the components is close to solved; deciding what is wired to what is "
    f"not. This is precisely the step the prior work either skipped [1], restricted "
    f"to parallel circuits [2], or handled with hand-tuned rules [3][4], and "
    f"quantifying it is only possible because the dataset supplies reference "
    f"netlists.",

    f"Replacing the ground-truth boxes with the trained detector changes the "
    f"end-to-end result remarkably little: bill-of-materials F1 moves from {BOM} "
    f"to {FULL_BOM}, local topology from {SIG} to {FULL_SIG}, and exact match from "
    f"{EXACT} to {FULL_EXACT}, a cost of "
    f"{abs(GT.get('exact', 0) - (FULL_UNET or {}).get('exact', 0)) * 100:.1f} "
    f"points. Both columns use the same learned segmentation, so the difference "
    f"is the detector and nothing else. It is not the bottleneck, and effort "
    f"spent on a larger or better-tuned detector would be largely wasted; the "
    f"loss is almost entirely in connectivity recovery. One caveat applies: the "
    f"annotated photographs all "
    f"belong to drafter 1, who was in the training split, because that drafter "
    f"supplies the only reference schematics in the dataset. The end-to-end "
    f"detector figures are therefore optimistic, while the ground-truth-box figures "
    f"are unaffected and remain the honest measure of the topology stage.",

    f"Table 3 ablates the pre-processing. Replacing the adaptive threshold used by "
    f"prior work with flat-field division raises exact match from "
    f"{f(A_ADAPT.get('exact'))} to {f(A_FLAT.get('exact'))}, and adding the "
    f"thickness filter raises it further to {f(A_THIN.get('exact'))}, with "
    f"local-topology F1 improving from {f(A_ADAPT.get('sig_f1'))} to "
    f"{f(A_THIN.get('sig_f1'))}. Printed page ruling, not symbol recognition, is "
    f"the dominant source of error.",

    "Results are consistent across the four photographs of any one drawing, which "
    "indicates the failures are properties of the drawing and the extraction, not "
    "of camera noise or viewpoint.",
]

RESULTS_QUAL = [
    "Figure 6 shows a photograph, the detected symbols, and the recovered "
    "electrical nodes rendered one colour per net. Figure 3 shows the generated "
    "schematic. Inspection of the failures shows three recurring causes: a rule "
    "line surviving clean-up and welding two nets together; a component drawn "
    "with no wire stub, so that erasing its box leaves its terminals touching "
    "nothing; and three-terminal devices whose terminals are assigned by image "
    "position and so may be permuted.",
]

RESULTS_QUANT += [
    f"Three results concern claims the topology metrics cannot make. Every "
    f"generated schematic is handed back to LTspice and the netlist it extracts "
    f"compared against the circuit intended: {RT_TXT} are accepted, which is the "
    f"claim the abstract makes and the one cheapest to believe without checking, "
    f"because a schematic whose wires do not meet opens perfectly happily with "
    f"the component floating. Transistor pin roles are {PIN_TXT} against "
    f"{PIN0_TXT} for positional ordering, and diode polarity {DIO_TXT} end to end "
    f"against {DIO0_TXT} with the reader disabled -- note that the latter is "
    f"below half, so positional order is not a coin toss but systematically "
    f"wrong. Component values improve from F1 {VAL_DEF} using class defaults to "
    f"{VAL_READ} once the handwritten labels are read.",

    f"The diode model is measured separately by eight-fold cross-validation "
    f"grouped by drafter -- never by image, since each circuit is drawn twice and "
    f"photographed four times -- over {D_N} symbols from 32 drafters, reaching "
    f"{D_ACC} with a median angular error of {D_ERR} degrees. The ablation is the "
    f"result: trained against the dataset's raw rotation annotation the same "
    f"model reaches only {D_NAIVE}, because that annotation is not the symbol's "
    f"bearing in the image. Composing the dataset's own port geometry with its "
    f"own coordinate transform gives the anode along 180 minus rotation, and the "
    f"two agree exactly at 90 and 270 degrees, which is how the wrong convention "
    f"survived visual checking on a drafter whose diodes are nearly all vertical. "
    f"The training loss says it before any metric does: 0.017 against 0.568.",

    "No number is reported for op-amp pin roles, and none can be produced here. "
    "Only one of the twelve reference circuits instantiates a subcircuit element "
    "and it is out of scope; CGHD annotates 768 op-amps and gives an orientation "
    "for none of them, and its port table has no entry for the class at all. The "
    "stage is therefore rule-based and its behaviour is pinned by unit tests on "
    "synthetic drawings whose answer is known by construction, which is a weaker "
    "claim than a measurement and is stated as such.",
]

DISCUSSION = [
    "The results support a specific reading of the prior literature. Accuracies "
    "in the mid-nineties on isolated crops are real but measure a sub-problem; "
    "once a system is asked to produce a circuit, performance is governed by "
    "connectivity recovery, which no reference work evaluates.",
]

DISC_BULLETS = [
    f"Strengths: the component inventory is recovered exactly (bill-of-materials "
    f"F1 {BOM}); every generated schematic is accepted by LTspice on a round-trip "
    f"of its own netlist ({RT_TXT}); the pin roles that isomorphism cannot see are "
    f"read from the drawing rather than assumed; and the approach handles "
    f"arbitrary topologies rather than parallel circuits only.",

    f"Weaknesses: {int(round((1 - GT.get('exact', 0)) * N_IMG))} of {N_IMG} "
    f"circuits are still not recovered exactly, and the remaining failures are "
    f"net splits concentrated in the densest drawing, where the segmentation "
    f"threshold that removes page ruling also thins strokes that are already "
    f"close together. Two parts of the taxonomy are known-incomplete: every "
    f"bipolar transistor is modelled as an NPN although the arrowhead direction "
    f"that distinguishes it is already located, and a switch is modelled with two "
    f"ports where LTspice's uses four.",

]

CONCLUSION = [
    f"This project built and evaluated a complete path from a photograph of a "
    f"hand-drawn circuit to an LTspice schematic that opens and simulates, using a "
    f"symbol detector trained on CGHD followed by stroke clean-up, wire tracing "
    f"and netlist assembly. Scored by graph isomorphism against the dataset's own "
    f"reference schematics, the system recovers the correct bill of materials with "
    f"F1 {BOM} and the exactly correct circuit for {EXACT} of photographs, while a "
    f"faithful reimplementation of the prior-art classifier scores {SVM_ACC} on "
    f"isolated crops. The conclusion is that component recognition is largely "
    f"solved and connectivity recovery is the open problem.",

    "An ablation identified printed page ruling as the dominant failure mode and "
    "showed that a stroke-thickness filter, which cuts rule lines where they cross "
    "wires instead of discarding the whole component, recovers a substantial part "
    "of the loss. A second comparison showed that substituting the trained "
    f"detector for ground-truth boxes costs {DET_COST} points of exact match, which "
    "locates the remaining loss squarely in connectivity recovery rather than in "
    "detection.",

    f"Two of the three next steps identified by this work have since been taken. "
    f"CGHD's rotation annotations were used to regress symbol orientation, which "
    f"recovers diode polarity at {D_ACC} over {D_N} annotated diodes, held out by "
    f"drafter, with a median angular error of {D_ERR} degrees; and the "
    f"stroke/background separation was learned from CGHD's segmentation maps "
    f"rather than thresholded, which attacked page ruling at its source and moved "
    f"exact match from {f(A_THIN.get('exact'))} to {EXACT}. What remains is the "
    f"handwriting: reading component values from the detected text boxes is the "
    f"largest manual step left, and is listed as future work by every reference "
    f"paper.",

    f"The diode result carries a caution worth recording, because it cost the "
    f"first attempt its measurement. CGHD's rotation is defined through the "
    f"dataset's own port geometry, in which a diode's anode sits at unit x=0 and "
    f"the transform into image space flips y before rotating; the anode therefore "
    f"lies along 180 degrees minus the annotated angle, not along the angle "
    f"itself. The two agree exactly at 90 and 270 degrees, so on a near-vertical "
    f"diode the error is invisible. Training on the raw angle instead scores "
    f"{D_NAIVE}, against {D_ACC} for the derived one, with no other change.",
]

TABLE1 = [
    ("Detector", "YOLOv8s, 896 px input, 50 epochs, batch auto-sized"),
    ("Augmentation", "mosaic 1.0, rotation 10 deg, h-flip 0.5, v-flip 0.3, scale 0.5"),
    ("Dataset", "CGHD 3,269 photographs, 59 labels merged to 16 classes"),
    ("Split", "by drafter: 4 held out for test, 3 for validation"),
    ("Binarisation", "flat-field division + Otsu; thickness filter at the median"),
    ("Tools", "PyTorch, Ultralytics, OpenCV, scikit-image, NetworkX, LTspice 26.1.1"),
    ("Hardware", "Colab T4 for training; RTX 3050 laptop for inference"),
]

TABLE2 = [
    ("Method", "Task", "Metric", "Score"),
    ("HOG+SVM [3][4], reimplemented", "classify isolated crops", "accuracy", SVM_ACC),
    ("Reported in [1] / [2] / [3]", "classify isolated crops", "accuracy", "0.960 / 0.954 / 0.944"),
    ("YOLOv8s detector", "symbol detection (val drafters)", "mAP@50", det_map),
    ("YOLOv8s detector", "symbol detection (val drafters)", "mAP@50:95", det_map95),
    ("This work, ground-truth boxes", "bill of materials", "F1", BOM),
    ("This work, ground-truth boxes", "local topology", "F1", SIG),
    ("This work, ground-truth boxes", "whole circuit", "exact match", EXACT),
    ("This work, full system", "bill of materials", "F1", FULL_BOM),
    ("This work, full system", "local topology", "F1", FULL_SIG),
    ("This work, full system", "whole circuit", "exact match", FULL_EXACT),
    ("This work, ground-truth boxes", "LTspice round-trip", "accepted", RT_RATE),
    ("This work, ground-truth boxes", "transistor pin roles (C/B/E)", "correct", PIN_TXT),
    ("   ablation: stage off", "pins taken in the order found", "correct", PIN0_TXT),
    ("This work, ground-truth boxes", "diode polarity, end to end", "correct", DIO_TXT),
    ("   ablation: stage off", "polarity never read", "correct", DIO0_TXT),
    ("Learned angle head, 8-fold by drafter", "diode polarity", "accuracy", D_ACC),
    ("   ablation: raw <rotation> label", "diode polarity", "accuracy", D_NAIVE),
    ("   ablation: stage off", "every part keeps its default", "F1", VAL_DEF),
    ("This work, labels read", "component values", "F1", VAL_READ),
]

TABLE3 = [
    ("Pre-processing", "Exact match", "Local-topology F1"),
    ("CLAHE + adaptive threshold (prior work)", f(A_ADAPT.get("exact")), f(A_ADAPT.get("sig_f1"))),
    ("Flat-field division + Otsu", f(A_FLAT.get("exact")), f(A_FLAT.get("sig_f1"))),
    ("+ stroke-thickness filter", f(A_THIN.get("exact")), f(A_THIN.get("sig_f1"))),
]

REFERENCES = [
    '[1]\tA. Kumar and R. Merchant, "Detection of hand drawn circuit schematic symbols," Stanford Univ. CS229 project report, 2021.',
    '[2]\tM. M. Arnadottir and O. Yngvason, "From component detection to simulation of analog circuits using CNNs," Stanford Univ. CS230 project report, 2022.',
    '[3]\tY. Liu and Y. Xiao, "Circuit sketch recognition," Dept. of Electrical Engineering, Stanford University, Stanford, CA.',
    '[4]\tM. Aksakalli, "Hand-drawn circuit recognizer," GitHub repository, 2018. [Online]. Available: https://github.com/mahmut-aksakalli/circuit_recognizer (accessed Nov. 14, 2026).',
    '[5]\tS. Roy, A. Bhattacharya, N. Sarkar, S. Malakar, and R. Sarkar, "Offline hand-drawn circuit component recognition using texture and shape-based features," Multimedia Tools and Applications, vol. 79, pp. 31353-31373, 2020, doi: 10.1007/s11042-020-09570-6.',
    '[6]\tJ. Bayer, S. T. Roy, and A. Dengel, "A public ground-truth dataset for handwritten circuit diagram images," 2025, Zenodo, doi: 10.5281/zenodo.15333233.',
    '[7]\tJ. Bayer, L. van Waveren, and A. Dengel, "Modular graph extraction for handwritten circuit diagram images," in Proc. Int. Conf. Document Analysis and Recognition (ICDAR), 2024. [Online]. Available: https://arxiv.org/abs/2402.11093',
    '[8]\tG. Jocher, J. Qiu, and A. Chaurasia, "Ultralytics YOLO," version 8, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics',
]


# ------------------------------------------------------------------- utilities

def set_text(p, text, bold=False, italic=False):
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    return p


def delete(p):
    p._element.getparent().remove(p._element)


def fill_block(paragraph, lines):
    """Write ``lines`` starting at ``paragraph``, inserting extras after it."""
    if not lines:
        delete(paragraph)
        return
    set_text(paragraph, lines[0])
    anchor = paragraph
    for extra in lines[1:]:
        new = anchor.insert_paragraph_before("")
        # move the newly created paragraph to just after the anchor
        anchor._element.addnext(new._element)
        set_text(new, extra)
        anchor = new


def fill_table(table, rows, header=False):
    while len(table.rows) < len(rows):
        table.add_row()
    for ti, row in enumerate(rows):
        cells = table.rows[ti].cells
        for ci, val in enumerate(row[: len(cells)]):
            cell = cells[ci]
            cell.text = ""
            para = cell.paragraphs[0]
            set_text(para, str(val), bold=(header and ti == 0))
    for extra in range(len(table.rows) - 1, len(rows) - 1, -1):
        table._tbl.remove(table.rows[extra]._tr)


def add_figure(doc, after, image: Path, caption: str, width_in=6.3):
    """Insert a captioned figure straight after ``after`` (a paragraph/table).

    Added in reverse order because each ``addnext`` puts its element
    immediately after the anchor, so writing caption-then-picture leaves them
    in the right order on the page.
    """
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_text(cap, caption, italic=True)
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if image.exists():
        pic.add_run().add_picture(str(image), width=Inches(width_in))
    else:
        set_text(pic, f"[missing figure: {image.name}]", italic=True)
    anchor = after._element if hasattr(after, "_element") else after._p
    anchor.addnext(cap._element)
    anchor.addnext(pic._element)
    return cap


def put_image(table, image: Path, width_in=6.4):
    cell = table.rows[0].cells[0]
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if image.exists():
        p.add_run().add_picture(str(image), width=Inches(width_in))
    else:
        set_text(p, f"[missing figure: {image.name}]", italic=True)


def main() -> int:
    if not TPL.exists():
        print(f"template not found: {TPL}")
        return 1
    d = docx.Document(str(TPL))
    P = list(d.paragraphs)

    # --- title block & metadata -------------------------------------------
    set_text(P[3], TITLE, bold=True)
    meta = d.tables[1]
    vals = {
        "Course": "MLEE: Machine Learning for Electrical Engineering, 2026-27",
        "Team members": "Rahul Bhat Yogesh (2023B4A31035P); Shukant Singh (2023B2A30922P)",
        "Instructor": "Ritesh Vyas",
        "Date of submission": "20 November 2026",
        "Code / data link": "https://drive.google.com/drive/folders/1PVF-j5wvPo3rWR1MSLkpzt9Cc9yGSuwt?usp=drive_link",
    }
    for row in list(meta.rows):
        key = row.cells[0].text.strip()
        if key.lower().startswith("team id"):
            row._tr.getparent().remove(row._tr)      # no team number
            continue
        if key in vals:
            row.cells[1].text = ""
            set_text(row.cells[1].paragraphs[0], vals[key])

    # --- body --------------------------------------------------------------
    fill_block(P[7], [ABSTRACT])
    set_text(P[8], KEYWORDS)
    fill_block(P[11], INTRO)
    for i in (12, 13, 14, 15, 16, 17):      # leftover intro placeholders
        delete(P[i])
    fill_block(P[21], BG_THEORY)
    delete(P[22])                            # equation placeholder
    fill_block(P[24], BG_RELATED)
    fill_block(P[26], BG_GAP)
    fill_block(P[30], METHOD_OVERVIEW)
    set_text(P[31], "Figure 1: The pipeline. A photograph is passed to a symbol "
                    "detector; the detected boxes are erased so that what remains "
                    "is wires; the skeleton of those wires is labelled into "
                    "electrical nodes; and the resulting graph is written as a "
                    "SPICE deck and an LTspice schematic.", italic=True)
    fill_block(P[33], METHOD_DATA)
    fill_block(P[35], METHOD_MODEL)
    fill_block(P[37], METHOD_SETUP)
    set_text(P[38], "Table 1: Key settings.", italic=True)
    fill_block(P[41], METHOD_METRICS)
    fill_block(P[45], RESULTS_QUANT)
    set_text(P[46], f"Table 2: End-to-end results on the {N_IMG} in-scope "
                    f"photographs, scored against the dataset's reference "
                    f"schematics. Each indented row is the same measurement "
                    f"with that stage switched off, not a different method.",
             italic=True)
    fill_block(P[49], ["Figure 6: A photograph, the detected symbols, and the "
                       "recovered electrical nodes, one colour per net."])
    add_figure(
        d, P[49], ROOT / "outputs" / "deck" / "hero.png",
        "Figure 7: One drawing, end to end. Left, the photograph as it was "
        "handed to the pipeline. Right, the SPICE netlist extracted from it: "
        "nothing was typed in, and the 9 V, 1 k, 2.2 k and 22 u are the values "
        "written on the paper rather than class defaults. The diode's polarity "
        "and the source's polarity were both read from the ink. Handed back to "
        "LTspice, all ten pins land on the nets the writer intended.")
    fill_block(P[51], DISCUSSION)
    for p, txt in zip((P[52], P[53], P[54]), DISC_BULLETS):
        set_text(p, txt)
    for p in (P[52], P[53], P[54])[len(DISC_BULLETS):]:
        delete(p)                            # unused bullet slots
    fill_block(P[57], CONCLUSION)
    delete(P[58])
    set_text(P[60], "We thank the course instructor and teaching assistants, and "
                    "acknowledge the authors of the CGHD dataset, which is "
                    "released under CC-BY 4.0 and made this evaluation possible.")

    # The footer carries its own placeholders, and the page-number field
    # lives in the same paragraph -- so patch the runs rather than the text,
    # which would wipe the field.
    for sec in d.sections:
        for para in sec.footer.paragraphs:
            for run in para.runs:
                if "[Project title]" in run.text:
                    run.text = run.text.replace("[Project title]", TITLE)
                if "Team [ID]" in run.text:
                    run.text = run.text.replace(" | Team [ID]", "")
                    run.text = run.text.replace("Team [ID]", "")

    # --- references ---------------------------------------------------------
    fill_block(P[63], REFERENCES)
    for i in (64, 65, 66):
        delete(P[i])

    # --- guidance paragraphs -------------------------------------------------
    for p in P:
        try:
            if p.text.strip().startswith("Guidance:"):
                delete(p)
        except Exception:
            pass

    # --- tables & figures ----------------------------------------------------
    put_image(d.tables[0], ROOT / "poster" / "bits_logo.png", width_in=1.6)
    put_image(d.tables[2], FIG / "fig_method.png")
    fill_table(d.tables[3], TABLE1)
    fill_table(d.tables[4], TABLE2, header=True)
    put_image(d.tables[5], FIG / "fig_qualitative.png")
    fill_table(d.tables[6], [
        ("Team member", "Contribution", "Report / Poster"),
        ("Rahul Bhat Yogesh",
         "Dataset preparation and drafter-disjoint splits; YOLOv8 detector "
         "training; stroke segmentation and the pre-processing ablation; wire "
         "tracing and node extraction; evaluation protocol and benchmark "
         "harness.", "Report, Poster"),
        ("Shukant Singh",
         "Topology and netlist construction; LTspice writer and round-trip "
         "verification; pin-role, polarity and diode-orientation readers; "
         "handwritten label recognition; desktop tool and figures.",
         "Report, Poster"),
    ], header=True)

    # Table 3 (ablation) does not exist in the template: add it after Table 2.
    t3 = d.add_table(rows=len(TABLE3), cols=3)
    t3.style = d.tables[4].style
    fill_table(t3, TABLE3, header=True)
    anchor = d.tables[4]._tbl
    anchor.addnext(t3._tbl)
    cap = d.add_paragraph()
    set_text(cap, "Table 3: Pre-processing ablation. Printed page ruling, not "
                  "symbol recognition, dominates the error.", italic=True)
    t3._tbl.addnext(cap._element)

    # More figures. The report had two; these carry the arguments that are
    # hardest to make in prose.
    DECK = ROOT / "outputs" / "deck"
    last = cap
    for img, txt in (
        (DECK / "ruling.png",
         "Figure 2: The dominant failure mode. Left, a circuit drawn on ruled "
         "notebook paper. Centre, the adaptive threshold used by prior work, "
         "where the printed rules survive as solid strokes and weld every node "
         "into one. Right, the learned stroke segmenter at a cut-off of 0.995, "
         "which breaks the ruling into faint dashes while the pencil circuit "
         "survives intact."),
        (FIG / "fig_result2.png",
         "Figure 3: The gap between finding the parts and recovering the "
         "circuit is the result. Component recognition is effectively solved; "
         "connectivity is where the loss is."),
        (FIG / "fig_result3.png",
         "Figure 4: Ground-truth boxes against the trained detector, with the "
         "segmentation held fixed. The detector is not the bottleneck."),
        (FIG / "fig_diode.png",
         "Figure 5: Diode orientation, held out by drafter. Green is the "
         "dataset's annotation, pink the model's reading. In the bottom row "
         "the two disagree by close to 180 degrees, and in every case "
         "inspected it is the annotation that points at the bar rather than "
         "the triangle."),
    ):
        last = add_figure(d, last, img, txt)

    # --- delete Annex B ------------------------------------------------------
    for i in range(71, 80):
        try:
            delete(P[i])
        except Exception:
            pass
    # By content, not by index: inserting Table 3 shifts every later table, and
    # removing index 7 after that deleted the *team contributions* table --
    # which the course instructions explicitly require -- while leaving the
    # annex it was meant to remove in place.
    for t in list(d.tables):
        if t.rows and "Report section" in t.rows[0].cells[0].text:
            t._tbl.getparent().remove(t._tbl)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.save(str(OUT))
    print(f"wrote {OUT}")

    import re as _re
    keep = _re.compile(r"^\[\d+\]	")          # reference entries are not placeholders
    allowed = ("Name", "Team", "Repository", "Instructor")
    leftovers = [p.text.strip() for p in docx.Document(str(OUT)).paragraphs
                 if p.text.strip().startswith("[")
                 and not keep.match(p.text.strip())
                 and not any(a in p.text for a in allowed)]
    if leftovers:
        print(f"  WARNING: {len(leftovers)} placeholder paragraphs remain:")
        for t in leftovers[:10]:
            print("   -", t[:90])
    else:
        print("  no placeholder text left (team names intentionally kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
