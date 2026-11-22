"""A desktop front end: pick a photograph, get an LTspice schematic open on screen.

The command line already does everything, but it asks you to know a weights
path, an output directory and a SPICE directive before you can see anything.
This is the demo form of the same pipeline -- choose an image, watch the stages
go by, and LTspice opens on the result.

    python -m circuit2spice.gui
    python -m circuit2spice.gui photo.jpg       # start with one already chosen

Everything slow happens on a worker thread, so the window keeps repainting
while YOLO and LTspice run. Tk is not thread-safe, so the worker never touches
a widget: it posts messages onto a queue that the UI drains on a timer.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

#: Where a trained detector usually lives, best first. Ground-truth boxes are
#: not an option here: this tool is for photographs that have no annotation.
WEIGHT_CANDIDATES = (
    "models/best.pt",
    "runs/yolov8s-2/weights/best.pt",
    "runs/yolov8s/weights/best.pt",
    "runs/yolov8s-3/weights/best.pt",
    "best.pt",
)

IMAGE_TYPES = [("Photographs", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
               ("All files", "*.*")]


def find_weights(explicit: str | None = None) -> Path | None:
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    env = os.environ.get("CIRCUIT2SPICE_WEIGHTS", "")
    if env and Path(env).is_file():
        return Path(env)
    for c in WEIGHT_CANDIDATES:
        if Path(c).is_file():
            return Path(c)
    return None


def find_diode_model() -> Path | None:
    """Where the diode polarity model lives, if it is installed at all.

    Reported at startup for the same reason the detector is: the pipeline
    falls back to positional terminal order without it, and a diode that is
    simply the wrong way round simulates perfectly happily. Silence here
    would be indistinguishable from success.
    """
    env = os.environ.get("CIRCUIT2SPICE_DIODE", "")
    if env and Path(env).is_file():
        return Path(env)
    p = Path("models/diode_orient.ts.pt")
    return p if p.is_file() else None


#: What the Corrections box calls each of ``process``'s override arguments.
_CORRECTION_KINDS = {"value": "values", "values": "values",
                     "as": "reclass", "type": "reclass",
                     "supply": "supplies", "supplies": "supplies",
                     "pins": "pins", "pin": "pins"}

#: Whitespace around ``=`` and ``,`` is closed up before the line is split.
#: Writing ``as R10 = inductor`` is the natural thing to do and used to parse
#: as nothing at all: three separate tokens, none of them a pair.
_TIDY = re.compile(r"\s*([=,])\s*")


def build_stamp() -> str:
    """Where the running code came from, and when it was last written.

    Printed at startup because a stale process is indistinguishable from a
    broken fix: the window stays open, Python keeps the module it already
    imported, and the old behaviour is reported as the new behaviour not
    working. Comparing this line against the file on disk settles it in one
    glance.
    """
    import datetime

    src = Path(__file__).resolve()
    try:
        when = datetime.datetime.fromtimestamp(src.stat().st_mtime)
        age = f", gui.py {when:%Y-%m-%d %H:%M}"
    except OSError:
        age = ""
    return f"{src.parent}{age}"


def parse_corrections(text: str, problems: list | None = None) -> dict:
    """One line of corrections into keyword arguments for ``process``.

    ``as V2=source_current value I1=1m supply VCC=5 pins M1=N001,N005,0``

    A bare word names which kind of correction the ``REF=VALUE`` items after
    it are, and ``value`` is assumed until one says otherwise, so the common
    case is just ``R1=10k R2=4k7``.

    **Leading dashes are stripped**, because the obvious thing to do with this
    box is paste the command line into it, and ``--as`` must not be read as
    anything other than ``as``. It used to be: an unknown bare word left the
    kind alone, so ``--as R10=inductor`` quietly became *value* ``R10 =
    "inductor"`` -- the part was never re-typed, the netlist read
    ``R10 N005 N002 inductor``, and nothing said a word about it.

    A word that is still not recognised is collected in ``problems`` rather
    than raised. This is a text box in a demo form and a half-typed word must
    not take the run down with it, but it must not be silently obeyed as
    something else either.
    """
    out: dict = {}
    kind = "values"
    # Whitespace is the only separator, so space around the ``=`` and the
    # commas is closed up first: ``as R10 = inductor`` and
    # ``pins M1 = N001, N005, 0`` are what people actually type, and splitting
    # them raw yields a pile of tokens with no pairs in it.
    for token in _TIDY.sub(r"\1", text or "").split():
        if "=" in token:
            ref, _, val = token.partition("=")
            if ref.strip() and val.strip():
                out.setdefault(kind, {})[ref.strip()] = val.strip()
            elif problems is not None:
                problems.append(token)
        else:
            word = token.lower().strip("-").rstrip(":")
            if word in _CORRECTION_KINDS:
                kind = _CORRECTION_KINDS[word]
            elif word and problems is not None:
                problems.append(token)
    return out


class App:
    def __init__(self, root: tk.Tk, image: str | None = None):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.image_path: Path | None = None
        self.result = None
        self.asc: Path | None = None
        self.busy = False
        self._previews: list = []          # keep PhotoImage refs alive

        root.title("circuit2spice — photo to LTspice")
        root.geometry("1040x720")
        root.minsize(860, 600)

        self._build()
        # Which build is actually running. Python loads a module once, so a
        # GUI left open across an edit keeps the old code and goes on
        # behaving the old way -- which looks exactly like the fix not
        # working, and cost two rounds of exactly that. The path says where
        # the code came from; the timestamp says how old it is.
        self.log(f"circuit2spice: {build_stamp()}")
        self.weights = find_weights()
        if self.weights:
            self.log(f"detector: {self.weights}")
        else:
            self.log("No detector weights found. Use 'Weights…' to pick a best.pt.")
            self.log("Looked in: " + ", ".join(WEIGHT_CANDIDATES))
        diode = find_diode_model()
        if diode:
            self.log(f"diode polarity: {diode}")
        else:
            self.log("diode polarity: no model installed — diodes will keep "
                     "positional terminal order. Build one with "
                     "`python scripts/diode_orient.py --build --train`, "
                     "or use Flip below.")
        self._poll()
        if image:
            self.set_image(Path(image))

    # ------------------------------------------------------------------ ui

    def _build(self) -> None:
        bar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        bar.pack(fill="x")

        self.btn_open = ttk.Button(bar, text="Choose photo…", command=self.choose)
        self.btn_open.pack(side="left")
        ttk.Button(bar, text="Weights…", command=self.choose_weights).pack(side="left", padx=(6, 0))

        self.btn_run = ttk.Button(bar, text="Convert  ▶", command=self.run, state="disabled")
        self.btn_run.pack(side="left", padx=(12, 0))

        self.btn_lt = ttk.Button(bar, text="Open in LTspice", command=self.open_ltspice,
                                 state="disabled")
        self.btn_lt.pack(side="left", padx=(6, 0))
        self.btn_dir = ttk.Button(bar, text="Show files", command=self.open_folder,
                                  state="disabled")
        self.btn_dir.pack(side="left", padx=(6, 0))

        opts = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        opts.pack(fill="x")
        self.v_sim = tk.BooleanVar(value=False)
        self.v_verify = tk.BooleanVar(value=True)
        self.v_autoopen = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Verify with LTspice", variable=self.v_verify).pack(side="left")
        ttk.Checkbutton(opts, text="Simulate", variable=self.v_sim).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(opts, text="Open LTspice when done",
                        variable=self.v_autoopen).pack(side="left", padx=(12, 0))
        ttk.Label(opts, text="   Analysis:").pack(side="left")
        self.v_analysis = tk.StringVar(value=".op")
        ttk.Entry(opts, textvariable=self.v_analysis, width=24).pack(side="left", padx=(4, 0))

        # The command line's --value/--as/--pins/--supply, in one box. Some of
        # what the pipeline gets wrong is not learnable from CGHD at all -- it
        # holds no current sources whatsoever -- so the corrections are not a
        # stopgap for a model that needs more training, and a demo form that
        # cannot reach them cannot produce a correct netlist for those
        # drawings at all.
        ov = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        ov.pack(fill="x")
        ttk.Label(ov, text="Corrections:").pack(side="left")
        self.v_over = tk.StringVar(value="")
        ttk.Entry(ov, textvariable=self.v_over).pack(
            side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Label(ov, text="  e.g.  as V2=source_current  value I1=1m  "
                           "supply VCC=5  pins M1=N001,N005,0").pack(side="left")

        # Polarity is the one thing the pipeline can get wrong in a way that
        # still simulates, so the demo form needs the override the command
        # line has as --flip. Re-exporting is cheap: the photograph is not
        # looked at again, only the netlist and the .asc are rewritten.
        pol = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        pol.pack(fill="x")
        ttk.Label(pol, text="Polarity:").pack(side="left")
        self.v_flip = tk.StringVar(value="")
        self.cb_flip = ttk.Combobox(pol, textvariable=self.v_flip, width=10,
                                    state="disabled", values=())
        self.cb_flip.pack(side="left", padx=(6, 0))
        self.btn_flip = ttk.Button(pol, text="Reverse this part",
                                   command=self.flip, state="disabled")
        self.btn_flip.pack(side="left", padx=(6, 0))
        self.lbl_flip = ttk.Label(pol, text="")
        self.lbl_flip.pack(side="left", padx=(10, 0))

        self.status = ttk.Label(self.root, text="Choose a photograph of a circuit.",
                                padding=(10, 0))
        self.status.pack(fill="x")
        self.bar = ttk.Progressbar(self.root, mode="determinate", maximum=100)
        self.bar.pack(fill="x", padx=10, pady=(4, 8))

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.nb = nb

        self.tab_log = ttk.Frame(nb); nb.add(self.tab_log, text="Progress")
        self.txt = tk.Text(self.tab_log, wrap="word", height=10, font=("Consolas", 10))
        sb = ttk.Scrollbar(self.tab_log, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.txt.pack(fill="both", expand=True)

        self.tab_net = ttk.Frame(nb); nb.add(self.tab_net, text="Netlist")
        self.net = tk.Text(self.tab_net, wrap="none", font=("Consolas", 10))
        self.net.pack(fill="both", expand=True)

        self.tab_img = ttk.Frame(nb); nb.add(self.tab_img, text="What it saw")
        self.canvas = tk.Canvas(self.tab_img, background="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def log(self, msg: str) -> None:
        self.txt.insert("end", msg.rstrip() + "\n")
        self.txt.see("end")

    def set_status(self, msg: str, pct: float | None = None) -> None:
        self.status.configure(text=msg)
        if pct is not None:
            self.bar["value"] = pct

    # --------------------------------------------------------------- actions

    def choose(self) -> None:
        p = filedialog.askopenfilename(title="Choose a photograph", filetypes=IMAGE_TYPES)
        if p:
            self.set_image(Path(p))

    def set_image(self, p: Path) -> None:
        self.image_path = p
        self.btn_run.configure(state="normal" if self.weights else "disabled")
        self.btn_lt.configure(state="disabled")
        self.btn_dir.configure(state="disabled")
        self.set_status(f"Ready: {p.name}", 0)
        self.log(f"\nphoto: {p}")
        if not self.weights:
            self.log("...but no detector weights are set, so Convert is disabled.")

    def choose_weights(self) -> None:
        p = filedialog.askopenfilename(title="Choose detector weights",
                                       filetypes=[("PyTorch weights", "*.pt"), ("All", "*.*")])
        if p:
            self.weights = Path(p)
            self.log(f"detector: {self.weights}")
            if self.image_path:
                self.btn_run.configure(state="normal")

    def run(self) -> None:
        if self.busy or not self.image_path or not self.weights:
            return
        self.busy = True
        for b in (self.btn_run, self.btn_open, self.btn_lt, self.btn_dir,
                  self.btn_flip):
            b.configure(state="disabled")
        self.cb_flip.configure(state="disabled", values=())
        self.lbl_flip.configure(text="")
        self.result = None
        self.net.delete("1.0", "end")
        self.nb.select(self.tab_log)
        # Read every Tk variable here, on the main thread, and hand the worker
        # plain values. Tk is not thread-safe: `self.v_analysis.get()` from a
        # worker is a cross-thread interpreter call that raises outright when
        # no mainloop happens to be running, and races when one is.
        # Parsed here, on the main thread, so what was *not* understood can be
        # said before the run rather than discovered in the netlist.
        bad: list = []
        typed = self.v_over.get().strip()
        corrections = parse_corrections(typed, problems=bad)
        for kind, items in corrections.items():
            self.log(f"corrections ({kind}): "
                     + "  ".join(f"{k}={v}" for k, v in items.items()))
        if bad:
            self.log(f"!! ignored in Corrections: {' '.join(bad)}")
        if typed and not corrections:
            # The worst case is a box full of text that did nothing, because
            # the netlist then looks like the corrections were applied and
            # rejected rather than never read. Say so where it cannot be
            # missed, not only in the log.
            self.log("!! nothing in the Corrections box could be understood; "
                     "expected  as REF=class  value REF=v  pins REF=n1,n2  "
                     "supply NET=v")
            self.set_status("Corrections box not understood -- see Progress", 0)
        t = threading.Thread(
            target=self._work, daemon=True,
            args=(self.v_analysis.get().strip() or ".op",
                  bool(self.v_verify.get()), bool(self.v_sim.get()),
                  corrections))
        t.start()

    def _render(self, res, method: str) -> list:
        """The netlist as the Netlist tab shows it.

        Diodes are spelled ``anode -> cathode`` rather than as a bare pair,
        because that order is the whole content of the polarity decision and
        a reader cannot otherwise tell which way round the part ended up.

        Op-amps get the same treatment for the same reason, and need it more:
        the netlist order is ``invin noninvin out``, three bare net names
        carry no hint of which is which, and swapping the two inputs turns
        negative feedback into positive -- a circuit that still simulates,
        straight into the rail.
        """
        g = res.graph
        how = ("learned stroke segmentation" if method == "unet"
               else f"{method} thresholding")
        lines = [f"{len(res.detections)} detections -> {len(g.components)} components, "
                 f"{g.n_nodes} nodes", f"strokes read by {how}", ""]
        for c in g.components:
            names = [g.node_name(n) if n >= 0 else "NC" for n in c.nodes]
            if c.cls == "diode" and len(names) == 2:
                nets = f"{names[0]} -> {names[1]}  (anode -> cathode)"
            elif c.cls == "opamp" and len(names) == 3:
                nets = (f"{names[0]} {names[1]} -> {names[2]}"
                        f"  (in- in+ -> out)")
            else:
                nets = " ".join(names)
            lines.append(f"  {c.ref:5} {c.cls:16} [{nets}] = {c.value}")
        if g.warnings:
            lines.append("")
            for w in g.warnings:
                lines.append(f"  note: {w}")
        return lines

    def _work(self, analysis: str, verify: bool, simulate: bool,
              corrections: dict | None = None) -> None:
        """Worker thread: never touches a widget, only posts to the queue.

        The settings arrive as plain values because reading a ``tk.Variable``
        here would be touching one.
        """
        try:
            post = lambda kind, **kw: self.q.put((kind, kw))
            post("status", msg="Loading detector…", pct=5)

            from .cli import resolve_method
            from .detect import YoloDetector
            from .pipeline import process

            method = resolve_method("auto")
            detector = YoloDetector(str(self.weights))
            post("status", msg="Detecting symbols…", pct=25)

            out_dir = Path("outputs/gui") / self.image_path.stem
            res = process(
                self.image_path, detector=detector, out_dir=out_dir,
                analysis=analysis, verify=verify, method=method,
                **(corrections or {}),
            )
            post("status", msg="Tracing wires and building the netlist…", pct=65)

            self.result = res
            lines = self._render(res, method)
            post("netlist", text="\n".join(lines))
            for ln in lines:
                post("log", msg=ln)
            post("diodes", refs=[c.ref for c in res.graph.components
                                 if c.cls == "diode"])

            if res.verification is not None:
                post("log", msg=f"LTspice check: {res.verification.summary()}")

            if simulate:
                post("status", msg="Simulating…", pct=85)
                from . import ltspice
                sim = ltspice.simulate(res.outputs["asc"])
                post("log", msg=("simulation: " + (str(sim["raw"]) if sim["raw"]
                                                   else "FAILED — see the .log")))

            post("previews",
                 det=str(res.outputs.get("detections_png", "")),
                 nodes=str(res.outputs.get("nodes_png", "")))
            post("done", asc=str(res.outputs["asc"]),
                 ok=bool(res.verification is None or res.verification.ok))
        except Exception:
            self.q.put(("error", {"text": traceback.format_exc()}))

    def flip(self) -> None:
        """Reverse one part's terminals and re-export, without re-detecting."""
        ref = self.v_flip.get().strip()
        if self.busy or not ref or self.result is None:
            return
        self.busy = True
        for b in (self.btn_run, self.btn_open, self.btn_flip):
            b.configure(state="disabled")
        threading.Thread(
            target=self._flip_work, daemon=True,
            args=(ref, self.v_analysis.get().strip() or ".op",
                  bool(self.v_verify.get()))).start()

    def _flip_work(self, ref: str, analysis: str = ".op",
                   verify: bool = False) -> None:
        """Worker: rewrite the netlist, the .asc and the JSON for one flip.

        The photograph is not looked at again. Only the node order changes, so
        re-running detection and wire tracing would cost half a minute to
        reach the same graph.
        """
        try:
            post = lambda kind, **kw: self.q.put((kind, kw))
            post("status", msg=f"Reversing {ref} and re-exporting…", pct=40)

            from .asc_writer import write_asc
            from .cli import resolve_method
            from .netlist import to_netlist

            res = self.result
            g = res.graph
            hit = [c for c in g.components if c.ref.upper() == ref.upper()]
            if not hit:
                post("log", msg=f"no part called {ref}")
                post("flipdone", ok=True)
                return
            for c in hit:
                c.nodes = list(reversed(c.nodes))
            g.warnings.append(f"reversed {hit[0].ref} by hand")

            asc = Path(res.outputs["asc"])
            write_asc(g, asc, analysis=analysis)
            Path(res.outputs["cir"]).write_text(
                to_netlist(g, title=asc.stem, analysis=analysis))
            Path(res.outputs["json"]).write_text(g.to_json())

            if verify:
                post("status", msg="Re-checking with LTspice…", pct=75)
                from .verify import verify_graph
                res.verification = verify_graph(g, asc, analysis=analysis)
                post("log", msg=f"LTspice check: {res.verification.summary()}")

            lines = self._render(res, resolve_method("auto"))
            post("netlist", text="\n".join(lines))
            post("log", msg=f"reversed {hit[0].ref}; rewrote {asc.name}")
            post("flipdone",
                 ok=bool(res.verification is None or res.verification.ok))
        except Exception:
            self.q.put(("error", {"text": traceback.format_exc()}))

    # ----------------------------------------------------------- ui pumping

    def _poll(self) -> None:
        try:
            while True:
                kind, kw = self.q.get_nowait()
                getattr(self, f"_on_{kind}")(**kw)
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    def _on_log(self, msg): self.log(msg)

    def _on_status(self, msg, pct=None): self.set_status(msg, pct)

    def _on_netlist(self, text):
        self.net.delete("1.0", "end")
        self.net.insert("1.0", text)

    def _on_diodes(self, refs):
        """Offer the parts whose polarity is worth second-guessing."""
        self.cb_flip.configure(values=tuple(refs))
        if refs:
            self.v_flip.set(refs[0])
            self.lbl_flip.configure(
                text=f"{len(refs)} diode(s); reverse one if it reads backwards")
        else:
            self.v_flip.set("")
            self.lbl_flip.configure(text="no diodes in this circuit")

    def _on_flipdone(self, ok):
        self.busy = False
        self.btn_open.configure(state="normal")
        self.btn_run.configure(state="normal")
        self.btn_flip.configure(state="normal" if self.cb_flip["values"] else "disabled")
        self.set_status(
            ("Reversed — schematic re-verified against LTspice." if ok
             else "Reversed, but LTspice disagreed (see Progress)."), 100)

    def _on_previews(self, det, nodes):
        self._previews.clear()
        self.canvas.delete("all")
        x = 10
        for path in (det, nodes):
            if not path or not Path(path).is_file():
                continue
            img = self._thumbnail(Path(path), 460)
            if img is None:
                continue
            self._previews.append(img)
            self.canvas.create_image(x, 10, image=img, anchor="nw")
            x += img.width() + 12

    def _thumbnail(self, path: Path, width: int):
        """Downscale with OpenCV and hand Tk a PNG it can read without PIL."""
        try:
            import cv2
            im = cv2.imread(str(path))
            if im is None:
                return None
            h, w = im.shape[:2]
            s = width / max(1, w)
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            tmp = path.with_name(path.stem + ".thumb.png")
            cv2.imwrite(str(tmp), im)
            return tk.PhotoImage(file=str(tmp))
        except Exception:
            return None

    def _on_done(self, asc, ok):
        self.asc = Path(asc)
        self.busy = False
        self.btn_open.configure(state="normal")
        self.btn_run.configure(state="normal")
        self.btn_lt.configure(state="normal")
        self.btn_dir.configure(state="normal")
        if self.cb_flip["values"]:
            self.cb_flip.configure(state="readonly")
            self.btn_flip.configure(state="normal")
        self.set_status(
            ("Done — schematic verified against LTspice." if ok
             else "Done, but LTspice disagreed with the netlist (see Progress)."), 100)
        self.log(f"schematic: {asc}")
        if self.v_autoopen.get():
            self.open_ltspice()

    def _on_error(self, text):
        self.busy = False
        self.btn_open.configure(state="normal")
        self.btn_run.configure(state="normal")
        if self.cb_flip["values"]:
            self.btn_flip.configure(state="normal")
        self.set_status("Failed — see Progress.", 0)
        self.log(text)
        messagebox.showerror("circuit2spice", text.strip().splitlines()[-1])

    # ------------------------------------------------------------- launching

    def open_ltspice(self) -> None:
        if not self.asc or not self.asc.is_file():
            return
        from . import ltspice
        exe = ltspice.find_exe()
        try:
            if exe:
                subprocess.Popen([str(exe), str(self.asc.resolve())])
                self.log(f"opened in {exe.name}")
            else:
                os.startfile(str(self.asc.resolve()))   # fall back to the association
                self.log("opened with the system default for .asc")
        except Exception as e:
            messagebox.showerror("circuit2spice", f"Could not open LTspice:\n{e}")

    def open_folder(self) -> None:
        if self.asc:
            try:
                os.startfile(str(self.asc.parent.resolve()))
            except Exception:
                pass


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root, argv[0] if argv else None)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
