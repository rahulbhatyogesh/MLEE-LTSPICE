"""Command line entry point.

    circuit2spice run photo.jpg -o outputs/demo --weights best.pt
    circuit2spice run photo.jpg -o outputs/demo --gt annotation.xml --verify
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


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="circuit2spice")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="convert one photo to an LTspice schematic")
    r.add_argument("image")
    r.add_argument("-o", "--out", default="outputs/run")
    r.add_argument("--weights", default=None, help="trained YOLO weights")
    r.add_argument("--gt", default=None, help="VOC annotation, used as a perfect detector")
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--analysis", default=".op", help="SPICE directive, e.g. '.tran 1m'")
    r.add_argument("--value", action="append", help="override a value, e.g. R1=4.7k")
    r.add_argument("--verify", action="store_true", help="check the .asc with LTspice")
    r.add_argument("--simulate", action="store_true", help="also run the simulation")

    a = ap.parse_args(argv)
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
        analysis=a.analysis, values=_parse_values(a.value), verify=a.verify,
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
