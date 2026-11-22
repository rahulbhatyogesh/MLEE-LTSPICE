"""Command line entry point.

    circuit2spice run photo.jpg -o outputs/demo --weights best.pt
    circuit2spice run photo.jpg -o outputs/demo --gt annotation.xml --verify
    circuit2spice gui                      # pick a photo, get LTspice open
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_values(pairs: list) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--value expects REF=VALUE, got {p!r}")
        ref, val = p.split("=", 1)
        out[ref.strip()] = val.strip()
    return out


def resolve_method(method: str) -> str:
    """Turn ``auto`` into a concrete method.

    Learned segmentation scores 0.778 exact match against thresholding's 0.500,
    so it is the default wherever its model is on disk -- but the model is not
    in the repo, and a missing file must degrade to the thresholding path
    rather than crash the tool.
    """
    if method != "auto":
        return method
    from .segment import available

    return "unet" if available() else "flatfield"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="circuit2spice")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="convert one photo to an LTspice schematic")
    r.add_argument("image")
    r.add_argument("-o", "--out", default="outputs/run")
    r.add_argument("--weights", default=None, help="trained YOLO weights")
    r.add_argument("--gt", default=None, help="VOC annotation, used as a perfect detector")
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--method", default="auto",
                   help="stroke extraction: auto (unet if its model is present, "
                        "else flatfield) | unet | flatfield | adaptive")
    r.add_argument("--unet-thr", type=float, default=None,
                   help="stroke probability cut-off; see segment.DEFAULT_THRESHOLD")
    r.add_argument("--analysis", default=".op", help="SPICE directive, e.g. '.tran 1m'")
    r.add_argument("--value", action="append", help="override a value, e.g. R1=4.7k")
    r.add_argument("--ocr", default="auto", choices=("auto", "structured", "free", "off"),
                   help="how to read the labels; structured scores a legal "
                        "vocabulary, free runs a recogniser loose")
    r.add_argument("--net", action="append", metavar="NODE=NAME",
                   help="name a net, e.g. 4=Vout, so it can be probed by name")
    r.add_argument("--flip", action="append", metavar="REF",
                   help="reverse a part's terminals, e.g. D1 -- diode "
                        "polarity is read from the drawing but can abstain, "
                        "and source polarity is not read at all")
    r.add_argument("--as", dest="retype", action="append", metavar="REF=CLASS",
                   help="correct a misdetected part, e.g. V2=source_current. "
                        "CGHD has no current sources at all, so the detector "
                        "cannot learn that one and this is the only fix")
    r.add_argument("--supply", action="append", metavar="NET=VALUE",
                   help="drive a labelled rail the drawing never sources, "
                        "e.g. VCC=5; added by hand and reported as such")
    r.add_argument("--pins", action="append", metavar="REF=N1,N2,N3",
                   help="state a part's node order outright, e.g. "
                        "M1=N001,N005,0 -- the only override that can express "
                        "an arbitrary permutation, which --flip cannot")
    r.add_argument("--verify", action="store_true", help="check the .asc with LTspice")
    r.add_argument("--simulate", action="store_true", help="also run the simulation")

    g = sub.add_parser("gui", help="desktop app: pick a photo, open it in LTspice")
    g.add_argument("image", nargs="?", default=None)

    a = ap.parse_args(argv)
    if a.cmd == "gui":
        from .gui import main as gui_main

        return gui_main([a.image] if a.image else [])
    if a.cmd != "run":
        ap.error("unknown command")

    if not a.weights and not a.gt:
        ap.error("pass --weights (trained model) or --gt (ground-truth boxes)")

    from .pipeline import process

    detector = None
    if a.weights:
        from .detect import YoloDetector
        detector = YoloDetector(a.weights, conf=a.conf)

    res = process(
        a.image, detector=detector, gt_xml=a.gt, out_dir=a.out,
        analysis=a.analysis, values=_parse_values(a.value),
        nets={int(k): v for k, v in (_parse_values(a.net) or {}).items()},
        flip=a.flip, pins=_parse_values(a.pins),
        reclass=_parse_values(a.retype),
        supplies=_parse_values(a.supply),
        read_values=a.ocr != "off", ocr_mode=a.ocr,
        verify=a.verify,
        method=resolve_method(a.method), unet_thr=a.unet_thr,
    )
    g = res.graph
    print(f"{len(res.detections)} detections -> {len(g.components)} components, "
          f"{g.n_nodes} nodes")
    for c in g.components:
        nets = " ".join(g.node_name(n) if n >= 0 else "NC" for n in c.nodes)
        print(f"  {c.ref:5} {c.cls:16} [{nets}] = {c.value}")
    for w in g.warnings:
        print(f"  warning: {w}")
    for k, v in res.outputs.items():
        print(f"  {k}: {v}")

    if res.verification is not None:
        print(f"  LTspice check: {res.verification.summary()}")
        if not res.verification.ok:
            return 2

    if a.simulate:
        from . import ltspice
        sim = ltspice.simulate(res.outputs["asc"])
        print(f"  simulation raw file: {sim['raw']}")
        if sim["raw"] is None:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
