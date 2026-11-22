"""Build a human-checkable inspection pack for chosen circuits.

For every photo it writes one wide image:

    photo + labelled component boxes | extracted electrical nodes

Component boxes are tagged with the SPICE name we gave them and the nodes we
think their terminals land on, and every node region is tagged with its node
name. So the picture can be read directly against the netlist: follow a wire,
check the colour, check the label.

Alongside each circuit goes a text report: our netlist, the reference netlist,
and both grouped net-by-net, which is where a disagreement is easiest to spot.

    python scripts/inspect_circuit.py --circuits C3 C2 C12
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import load_references, predicted_elements
from circuit2spice.detect import detections_from_voc
from circuit2spice.netcompare import compare
from circuit2spice.pipeline import _load_and_scale
from circuit2spice.topology import build_circuit
from circuit2spice.wires import colorize

ROOT = Path("data/cghd/drafter_1")
OUT = Path("outputs/inspect")


def annotate(img, dets, graph):
    """Photo with every component boxed, named, and tagged with its nodes."""
    out = img.copy()
    thick = max(2, int(round(min(out.shape[:2]) / 500)))
    fs = max(0.5, min(out.shape[:2]) / 1400)
    by_box = {(round(c.det.cx), round(c.det.cy)): c for c in graph.components}
    for d in dets:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        comp = by_box.get((round(d.cx), round(d.cy)))
        if comp is not None:
            cv2.rectangle(out, p1, p2, (0, 200, 0), thick)
            nets = " ".join(graph.node_name(n) if n >= 0 else "NC" for n in comp.nodes)
            cv2.putText(out, comp.ref, (p1[0], max(14, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (0, 140, 0), thick)
            cv2.putText(out, nets, (p1[0], p2[1] + int(22 * fs * 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (200, 0, 0), thick)
        elif d.cls in ("junction", "crossover", "gnd", "terminal"):
            col = {"junction": (0, 0, 255), "crossover": (255, 0, 255),
                   "gnd": (0, 160, 255), "terminal": (160, 160, 0)}[d.cls]
            cv2.rectangle(out, p1, p2, col, thick)
            cv2.putText(out, d.cls[:4], (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, col, max(1, thick - 1))
    return out


def label_nodes(node_map, graph, shape):
    """Colour map with each node's name printed on it."""
    out = colorize(node_map)
    thick = max(2, int(round(min(shape[:2]) / 500)))
    fs = max(0.5, min(shape[:2]) / 1200)
    used = {n for c in graph.components for n in c.nodes if n >= 0}
    for node in sorted(used):
        ys, xs = np.nonzero(node_map.labels == node)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        name = graph.node_name(node)
        cv2.putText(out, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 1.2, (0, 0, 0), thick + 3)
        cv2.putText(out, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    fs * 1.2, (255, 255, 255), thick)
    return out


def nets_of(netlist: dict) -> dict:
    """net name -> ['R1.0', 'C2.1', ...]"""
    g = collections.defaultdict(list)
    for ref, nodes in netlist.items():
        for i, n in enumerate(nodes):
            g[n].append(f"{ref}.{i}")
    return dict(g)


def report(circuit, ref, results) -> str:
    L = [f"CIRCUIT {circuit}", "=" * 70, "",
         "REFERENCE netlist (what the dataset says the circuit is):"]
    for k, v in ref.items():
        L.append(f"   {k:5} {v}")
    L.append("")
    L.append("REFERENCE nets (which terminals share a wire):")
    for net, pins in sorted(nets_of(ref).items(), key=lambda t: -len(t[1])):
        L.append(f"   {net:6} ({len(pins)} pins): {', '.join(sorted(pins))}")
    L.append("")
    L.append("-" * 70)
    for stem, pred, ok in results:
        L.append("")
        L.append(f"{stem}   -> {'MATCH' if ok else 'MISMATCH'}")
        L.append("  our netlist:")
        for k, v in pred.items():
            L.append(f"     {k:5} {v}")
        L.append("  our nets:")
        for net, pins in sorted(nets_of(pred).items(), key=lambda t: -len(t[1])):
            L.append(f"     {net:6} ({len(pins)} pins): {', '.join(sorted(pins))}")
        rs = sorted((len(p) for p in nets_of(ref).values()), reverse=True)
        ps = sorted((len(p) for p in nets_of(pred).values()), reverse=True)
        L.append(f"  net sizes  reference={rs}")
        L.append(f"             ours     ={ps}")
        if rs != ps:
            L.append("  >> net sizes differ: a wire is split or merged somewhere")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", nargs="+", default=["C3", "C2", "C12"])
    ap.add_argument("--photos", type=int, default=8)
    a = ap.parse_args()

    refs = load_references()
    OUT.mkdir(parents=True, exist_ok=True)
    index = ["INSPECTION PACK", "=" * 70, "",
             "Each <stem>.png is: photo with named components | extracted nodes.",
             "Green box = component, with its SPICE name above and the nodes we",
             "think its terminals reach below. Red = junction, magenta = crossover,",
             "orange = ground. Right panel colours one electrical node per colour,",
             "with the node name printed on it.",
             "",
             "To check a circuit: pick a wire in the photo, see which components it",
             "touches, then confirm the right panel gives them all the same node name.",
             ""]

    for c in a.circuits:
        if c not in refs:
            print(f"  no reference for {c}, skipping"); continue
        d = OUT / c
        d.mkdir(parents=True, exist_ok=True)
        results = []
        for xml in sorted((ROOT / "annotations").glob(f"{c}_*.xml"))[: a.photos]:
            img_p = ROOT / "images" / f"{xml.stem}.jpg"
            if not img_p.exists():
                continue
            dets = detections_from_voc(xml)
            img, dets = _load_and_scale(img_p, dets, 1600)
            g, nm = build_circuit(img, dets)
            pred = predicted_elements(g)
            ok = compare(xml.stem, c, pred, refs[c],
                         pin_ground=any(x.cls == "gnd" for x in dets)).exact
            results.append((xml.stem, pred, ok))

            left = annotate(img, dets, g)
            right = label_nodes(nm, g, img.shape)
            pair = np.hstack([left, right])
            tag = "MATCH" if ok else "MISMATCH"
            cv2.putText(pair, f"{xml.stem}  {tag}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 150, 0) if ok else (0, 0, 220), 3)
            cv2.imwrite(str(d / f"{xml.stem}_{tag}.png"), pair)

        (d / f"{c}_netlists.txt").write_text(report(c, refs[c], results), encoding="utf-8")
        n_ok = sum(r[2] for r in results)
        index.append(f"{c}: {n_ok}/{len(results)} correct   -> {c}/")
        print(f"  {c}: {n_ok}/{len(results)} correct, wrote {len(results)} images + netlists")

    (OUT / "README.txt").write_text("\n".join(index), encoding="utf-8")
    print(f"\npack written to {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
