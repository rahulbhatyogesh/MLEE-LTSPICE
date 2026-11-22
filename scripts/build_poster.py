"""Fill the MLEE A0 tikzposter template.

Keeps the template's preamble (colours, title style, helper macros) verbatim and
replaces the body. Numbers are read from the same JSON files the report uses, so
the two cannot disagree -- which the brief explicitly checks for.

    python scripts/build_poster.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "poster" / "MLEE_Poster_Template.tex"
OUT = ROOT / "poster" / "MLEE_Poster.tex"
FIG = ROOT / "outputs" / "figures"

# tikzposter typesets the title in LR mode, so it neither wraps nor accepts a
# manual \ break; the parbox substitution below is what makes it wrap.
TITLE = "Hand-Drawn Circuit Schematics to LTspice Netlists"


def load(name, default=None):
    p = ROOT / "outputs" / name
    return json.loads(p.read_text()) if p.exists() else (default or {})


BENCH, ABL, SVM = load("benchmark.json"), load("ablation_preproc.json"), load("baseline_svm.json")
DET = load("detector.json", {})
FULL_UNET = load("benchmark_unet_full.json", {}).get("full-system", {})
DIODE = load("diode_orient.json", {})
GT = BENCH.get("gt-boxes", {})
TODO = r"\textit{pending}"


def f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else TODO


EXACT, SIG, BOM = f(GT.get("exact")), f(GT.get("sig_f1")), f(GT.get("bom_f1"))
SVM_ACC = f(SVM.get("accuracy"))
A_AD = ABL.get("adaptive|none|thinNone", {})
A_FL = ABL.get("flatfield|none|thinNone", {})
A_TH = ABL.get("flatfield|mask|thin50.0", {})
DET_MAP = f(DET.get("map50")) if DET else TODO
D_ACC, D_N = f(DIODE.get("accuracy")), str(DIODE.get("n", "?"))
RT = load("roundtrip.json", {}).get("gt-boxes", {})
RT_TXT = f"{RT['ok']}/{RT['images']}" if RT else TODO
_p = load("pins.json", {}).get("arrowhead/gt-boxes", {})
_p0 = load("pins.json", {}).get("positional/gt-boxes", {})
PIN_TXT = f"{_p['pins_correct']}/{_p['pins_scored']}" if _p else TODO
PIN0_TXT = f"{_p0['pins_correct']}/{_p0['pins_scored']}" if _p0 else TODO
_d = load("diodes.json", {}).get("learned/gt-boxes", {})
_d0 = load("diodes.json", {}).get("blind/gt-boxes", {})
DIO_TXT = f"{_d['diodes_correct']}/{_d['diodes_scored']}" if _d else TODO
DIO0_TXT = f"{_d0['diodes_correct']}/{_d0['diodes_scored']}" if _d0 else TODO
_v = load("values.json", {}).get("gt-boxes", {}).get("informative", {})
VAL_READ, VAL_DEF = f(_v.get("read")), f(_v.get("default"))
N_IMG = str(GT.get("images", 64))


BODY = r"""
\begin{document}
\maketitle

% =====================================================================
\begin{columns}
% ------------------------- LEFT COLUMN -------------------------------
\column{0.5}

\block{Summary}{
  Turning a hand-drawn circuit into a simulation is still done by hand. Prior
  work recognises \emph{components} well---94--96\%---but stops before a usable
  circuit, and the one published attempt at simulation was limited to purely
  parallel circuits. We build the full chain from photograph to a
  \textbf{working LTspice schematic}, and score it by graph isomorphism against
  reference netlists. Components are recovered at \textbf{F1 __BOM__}; the
  exactly correct circuit at only \textbf{__EXACT__}. \textbf{Finding the parts
  is solved; wiring them up is not.}
}

\block{Introduction and Objectives}{
  \textbf{Motivation:} engineers sketch circuits on paper long before they draw
  them in software. Retyping every component and connection into a schematic
  editor is slow and error-prone; a photograph should be enough.
  \vspace{3mm}

  \textbf{Objectives:}
  \begin{itemize}
    \item Train a symbol detector on CGHD---the labelled bounding-box dataset
          prior work said \textit{``was not to be found on the web''}.
    \item Recover connectivity for \textbf{arbitrary} topologies, not just the
          parallel circuits prior work was restricted to.
    \item Evaluate the output \textbf{as a circuit}, by graph isomorphism
          against reference schematics---not as a set of labels.
  \end{itemize}
}

\block{Background}{
  \textbf{Key idea:} a circuit is a labelled bipartite graph of components and
  electrical nodes. Two extractions describe the same circuit when their graphs
  are isomorphic under a type-preserving map:
  \[ G=(C \cup N,\;E), \qquad
     \hat{G} \cong G^{*} \iff \exists\,\phi:\; \text{type}(\phi(c))=\text{type}(c) \]
  \begin{itemize}
    \item MobileNet on isolated crops reaches 96\%, but grid and sliding-window
          scanning of full schematics \textit{``performed poorly''} [1]
    \item A CNN plus sliding windows reaches 79.4\% box accuracy and emits a
          netlist \textbf{for parallel circuits only}---general node detection
          \textit{``proved to be a daunting task''} [2]
    \item \textbf{Gap:} nobody evaluates whether the \emph{recovered circuit} is
          correct. We do, using CGHD's reference schematics [4].
  \end{itemize}
}

\block{Methodology}{
  \figslot{fig_method.png}{0.93\linewidth}{13cm}%
    {Figure placeholder\\Pipeline / block diagram of the approach\\(save as fig\_method.png)}
  \figcap{Figure 1: Detected symbols are \emph{erased} from the binarised image;
  what remains is wires. Their skeleton is labelled into electrical nodes.}
  \begin{enumerate}
    \item \textbf{Data:} CGHD [4], 3{,}269 photographs, 248k boxes, 59 labels
          merged to 16; split \textbf{by drafter}, never by image.
    \item \textbf{Detect:} YOLOv8s @ 896\,px, 50 epochs.
    \item \textbf{Clean:} flat-field division + Otsu, then a
          \textbf{stroke-thickness filter} to strip printed page ruling.
    \item \textbf{Trace:} skeleton $\rightarrow$ connected components
          $\rightarrow$ nodes; junctions union, crossovers re-link collinear stubs.
    \item \textbf{Order the pins:} isomorphism cannot see a reversed part, so
          the base is read from the bar, the emitter from the arrowhead, a
          gate from the lone lead, and a diode's polarity from a learned angle
          head. Each \emph{declines} when the drawing does not say.
    \item \textbf{Read the labels:} closed-vocabulary scoring, then a local
          vision-language model: component values F1 __VALD__ $\rightarrow$ __VALR__.
    \item \textbf{Write:} SPICE deck + LTspice \texttt{.asc}.
  \end{enumerate}
}

\block{Experimental Setup}{
  \begin{center}
  \begin{tabular}{ll}
    \toprule
    \textbf{Parameter} & \textbf{Value / description} \\
    \midrule
    Dataset        & CGHD, 3{,}269 photographs, 16 merged classes \\
    Split          & by drafter: 4 test, 3 validation \\
    Detector       & YOLOv8s, 896\,px, 50 epochs, mosaic + flips \\
    Evaluation     & __NIMG__ in-scope photographs, reference schematics \\
    Tools          & PyTorch, Ultralytics, OpenCV, NetworkX, LTspice 26.1.1 \\
    \bottomrule
  \end{tabular}
  \end{center}
}


% ------------------------- RIGHT COLUMN ------------------------------

\column{0.5}

\block{Results}{
  \figslot{fig_qualitative.png}{0.97\linewidth}{5.5cm}%
    {Figure placeholder: photograph, detections, nets}
  \figcap{Figure 2: Photograph, detected symbols, and the electrical nodes
  recovered from what is left once they are erased---one colour per net.}

  \begin{center}
  \begin{tabular}{lll}
    \toprule
    \textbf{Method} & \textbf{Task} & \textbf{Score} \\
    \midrule
    HOG+SVM [3], reimplemented & isolated crops   & __SVMACC__ \\
    Reported in [1] / [2]      & isolated crops   & 0.960 / 0.954 \\
    \midrule
    YOLOv8s detector & mAP@50 (val drafters) & __DETMAP__ \\
    \midrule
    This work & bill of materials (F1) & __BOM__ \\
    This work & local topology (F1)    & __SIG__ \\
    \textbf{This work} & \textbf{whole circuit (exact)} & \textbf{__EXACT__} \\
    \midrule
    \textbf{This work} & \textbf{LTspice accepts the .asc} & \textbf{__RT__} \\
    This work & transistor pins C/B/E & __PIN__ \\
    \quad\textit{ablation: stage off} & \textit{pins in found order} & \textit{__PIN0__} \\
    This work & diode polarity, end to end & __DIO__ \\
    \quad\textit{ablation: stage off} & \textit{polarity never read} & \textit{__DIO0__} \\
    This work & component values (F1) & __VALR__ \\
    \quad\textit{ablation: stage off} & \textit{parts keep their defaults} & \textit{__VALD__} \\
    \bottomrule
  \end{tabular}
  \end{center}
  \figcap{Table 1: __NIMG__ in-scope photographs, scored by graph isomorphism.
  The lower half is what isomorphism \emph{cannot} see, each row paired with the
  same measurement taken with that stage off. Note __DIO0__: positional order is
  not a coin toss.}

  \figslot{fig_result3.png}{0.90\linewidth}{5.2cm}%
    {Figure placeholder\\Ground-truth boxes vs trained detector\\(save as fig\_result3.png)}
  \figcap{Figure 3: Perfect boxes to the trained detector costs __DETCOST__
  points, segmentation held fixed. \textbf{The detector is not the bottleneck.}}

}

\block{Discussion}{
  \begin{itemize}
    \item \textbf{Finding:} mid-nineties accuracy on isolated crops is real but
          measures a \emph{sub-problem}. Asked to produce a circuit, performance
          is governed by connectivity, which no reference work evaluates.
    \item \textbf{Why ruling matters:} equalising local contrast promotes a
          printed rule line to the status of a pencil stroke, and one such line
          welds every node into one. Removing it \emph{by component} fails:
          where it crosses a wire, the two are one component. \textbf{Thickness
          cuts the crossing instead} (ruling 1--2\,px, pencil 3--6\,px).
    \item \textbf{Strength:} every schematic passing verification opens and
          simulates in LTspice; arbitrary topologies, not parallel only.
    \item \textbf{Limitation:} only one drafter ships reference schematics, and
          that drafter was in the detector's training split. Op-amp pins cannot be
          scored at all: CGHD gives an orientation for none of its 768.
  \end{itemize}
  \vspace{2mm}
  \begin{center}
  \begin{tabular}{lcc}
    \toprule
    \textbf{Pre-processing} & \textbf{Exact} & \textbf{Local F1} \\
    \midrule
    CLAHE + adaptive threshold \textit{(prior work)} & __AD_E__ & __AD_S__ \\
    Flat-field division + Otsu                       & __FL_E__ & __FL_S__ \\
    \textbf{+ stroke-thickness filter}               & \textbf{__TH_E__} & \textbf{__TH_S__} \\
    \bottomrule
  \end{tabular}
  \end{center}
  \figcap{Table 2: Printed page ruling, not symbol recognition, dominates the error.}
}

\block{Conclusion and Future Work}{
  \begin{itemize}
    \item Complete photograph $\rightarrow$ LTspice path; components at
          \textbf{F1 __BOM__}, exact circuit at \textbf{__EXACT__}.
    \item Component recognition is largely solved; \textbf{connectivity recovery
          is the open problem}.
    \item Symbol orientation \emph{is} regressed from CGHD's rotation
          annotations: diode polarity \textbf{__DACC__} over __DN__ diodes,
          held out by drafter.
    \item \textbf{Future work:} a per-image segmentation threshold for the
          densest drawings; four-terminal switches; PNP against NPN, for which
          the arrowhead is already located and unused.
  \end{itemize}
}



\begin{subcolumns}
\subcolumn{0.62}
\block{References}{
  {\footnotesize
  \begin{enumerate}[label={[\arabic*]}, leftmargin=1.9em, itemsep=1pt]
    \item A. Kumar and R. Merchant, ``Detection of hand drawn circuit schematic symbols,'' Stanford CS229 project report, 2021.
    \item M. M. \'Arnad\'ottir and \'O. Yngvason, ``From component detection to simulation of analog circuits using CNNs,'' Stanford CS230 project report, 2022.
    \item M. Aksakalli, ``Hand-drawn circuit recognizer,'' GitHub, 2018.
    \item J. Bayer \emph{et al.}, ``A public ground-truth dataset for handwritten circuit diagram images,'' Zenodo, 2025, doi:10.5281/zenodo.15333233.
    \item J. Bayer, L. van Waveren, A. Dengel, ``Modular graph extraction for handwritten circuit diagram images,'' \emph{Proc.\ ICDAR}, 2024.
  \end{enumerate}}
}
\subcolumn{0.38}
\block{Report and Code}{
  {\centering
  \IfFileExists{\qrfile}{%
    \includegraphics[width=3cm]{\qrfile}%
  }{%
    \begin{tikzpicture}
      \node[draw=guidegrey, dashed, line width=2pt, minimum width=4cm,
            minimum height=4cm, text=guidegrey, align=center,
            font=\large\itshape] {QR code\\(report / repo)};
    \end{tikzpicture}%
  }\par}\vspace{0.6em}
  {\footnotesize Acknowledgements: instructor, TAs, and the authors
  of the CGHD dataset (CC-BY 4.0).\par}
}
\end{subcolumns}

\end{columns}
\end{document}
"""


def main() -> int:
    if not TPL.exists():
        print(f"template not found: {TPL}")
        return 1

    tpl = TPL.read_text(encoding="utf-8")
    preamble = tpl.split(r"\begin{document}")[0]

    # Title / authors / affiliation.
    preamble = preamble.replace(
        r"\title{[Project Title: Concise, Specific and Informative]}",
        "\\title{" + TITLE + "}")
    preamble = preamble.replace(
        r"\author{[Student Name 1] \;\textbullet\; [Student Name 2] \;\textbullet\; [Student Name 3]"
        "\n        \\;\\;|\\;\\; Team [ID]}",
        r"\author{Rahul Bhat Yogesh (2023B4A31035P) \;\textbullet\; "
        r"Shukant Singh (2023B2A30922P)}")
    # tikzposter typesets the title in LR mode: it will not wrap, and a
    # manual break raises 'Not allowed in LR mode'.  Measured box width is
    # 1449pt, so the title has to fit one line at the chosen size.
    preamble = preamble.replace(
        r"{\bfseries\fontsize{64}{72}\selectfont \@title \par}",
        r"{\bfseries\fontsize{54}{64}\selectfont \@title \par}")
    preamble = preamble.replace("[Semester, Year]", "Semester I, 2026--27")
    preamble = preamble.replace(
        r"\;\textbullet\; MLEE \;\textbullet\; Instructor: [Name] \;\textbullet\;",
        r"\;\textbullet\; \mbox{Machine Learning for Electrical Engineering}"
        r" \;\textbullet\; Instructor: Ritesh Vyas \;\textbullet\;")
    # Ten blocks over two columns: 15mm between each is 135mm of the page
    # spent on gaps, which is what pushed References off the bottom.
    preamble = preamble.replace("blockverticalspace=15mm", "blockverticalspace=10mm")

    body = BODY
    for k, v in {
        "__BOM__": BOM, "__EXACT__": EXACT, "__SIG__": SIG, "__SVMACC__": SVM_ACC,
        "__DACC__": D_ACC, "__DN__": D_N,
        "__AD_E__": f(A_AD.get("exact")), "__FL_E__": f(A_FL.get("exact")),
        "__TH_E__": f(A_TH.get("exact")), "__DETMAP__": DET_MAP,
        "__AD_S__": f(A_AD.get("sig_f1")), "__FL_S__": f(A_FL.get("sig_f1")),
        "__TH_S__": f(A_TH.get("sig_f1")),
        "__RT__": RT_TXT, "__PIN__": PIN_TXT, "__PIN0__": PIN0_TXT,
        "__DIO__": DIO_TXT, "__DIO0__": DIO0_TXT,
        "__VALR__": VAL_READ, "__VALD__": VAL_DEF, "__NIMG__": N_IMG,
        "__DETCOST__": f(abs(GT.get("exact", 0)
                             - (FULL_UNET or {}).get("exact", 0)) * 100, 1),
    }.items():
        body = body.replace(k, v)

    OUT.write_text(preamble + body.lstrip("\n"), encoding="utf-8")
    print(f"wrote {OUT}")

    # Figures must sit next to the .tex for \includegraphics to find them.
    copied = []
    for name in ("fig_method.png", "fig_result1.png", "fig_result2.png",
                 "fig_result3.png", "fig_qualitative.png", "fig_schematic.png"):
        src = FIG / name
        if src.exists():
            shutil.copy2(src, OUT.parent / name)
            copied.append(name)
    print(f"  copied {len(copied)} figures next to the poster: {', '.join(copied)}")

    missing = [n for n in ("bits_logo.png", "dept_logo.jpg")
               if not (OUT.parent / n).exists()]
    if missing:
        print(f"  NOTE: add logo files to poster/ for the title block: {missing}")

    leftover = [tok for tok in ("__BOM__", "__EXACT__", "__SIG__", "__AD_E__",
                               "__DACC__", "__DN__")
                if tok in OUT.read_text(encoding="utf-8")]
    print("  unresolved tokens:", leftover or "none")
    # Match the actual block, not the preamble comment that names it.
    if "\block{Poster Guidelines" in OUT.read_text(encoding="utf-8"):
        print("  WARNING: guidelines block still present")
    else:
        print("  guidelines block removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
