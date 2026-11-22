"""Should the thickness filter run on the probability map rather than the mask?

The 0.995 cut-off works by discarding everything the network is unsure about.
Page ruling lives there -- but so do the faint strokes of a densely drawn
circuit, which is why C8 fell from 6/8 to 4/8 when learned segmentation went
in. Thickness is an independent signal: ruling is thin *and* uncertain, while
a faint pencil line is thin but part of a stroke that is thick somewhere. So
keep a pixel if the network is confident, or if it is merely plausible and
belongs to a thick stroke:

    mask = (p > thr) | thick(p > low, q)

``low = None`` is the shipped system, so the sweep contains it as a special
case and cannot flatter the hybrid by comparing against something else.

    python scripts/ablate_probthin.py
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import load_references, predicted_elements, scope, summarise
from circuit2spice import segment
from circuit2spice.detect import detections_from_voc
from circuit2spice.netcompare import compare
from circuit2spice.pipeline import _load_and_scale
from circuit2spice.topology import build_circuit

ROOT = Path("data/cghd/drafter_1")

#: (low cut-off, thickness percentile); None = the shipped single threshold.
SETTINGS = [(None, None), (0.97, 70.0), (0.9, 70.0), (0.9, 80.0), (0.5, 70.0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ablation_probthin.json")
    a = ap.parse_args()

    refs = load_references()
    in_scope = scope(refs)
    rows = {s: [] for s in SETTINGS}
    t0 = time.perf_counter()

    xmls = sorted(ROOT.glob("annotations/*.xml"))
    for n, xml in enumerate(xmls, 1):
        circuit = xml.stem.split("_")[0]
        img_p = ROOT / "images" / f"{xml.stem}.jpg"
        if circuit not in refs or not img_p.exists():
            continue
        dets = detections_from_voc(xml)
        img, dets = _load_and_scale(img_p, dets, 1600)
        has_gnd = any(d.cls == "gnd" for d in dets)
        for low, q in SETTINGS:
            mask = segment.segment_strokes(img, low_threshold=low, thin_q=q)
            g, _ = build_circuit(img, dets, bridge_px=6, method="unet", binary_override=mask) \
                if False else build_circuit(
                    img, dets, bridge_px=6, method="unet",
                    node_map=_node_map(img, dets, mask))
            rows[(low, q)].append(compare(xml.stem, circuit, predicted_elements(g),
                                          refs[circuit], pin_ground=has_gnd))
        if n % 24 == 0:
            print(f"  {n}/{len(xmls)}  {time.perf_counter()-t0:.0f}s", flush=True)

    out = {}
    for low, q in SETTINGS:
        keep = [r for r in rows[(low, q)] if in_scope.get(r.circuit, True)]
        label = "single threshold (shipped)" if low is None else f"+ thick(p>{low}, q={q:.0f})"
        key = "single" if low is None else f"low{low}|q{q:.0f}"
        out[key] = summarise(keep, f"{label:30}")
        per = collections.defaultdict(list)
        for r in keep:
            per[r.circuit].append(r.exact)
        out[key]["per_circuit"] = {c: f"{sum(v)}/{len(v)}" for c, v in
                                   sorted(per.items(), key=lambda t: int(t[0][1:]))}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    return 0


def _node_map(img, dets, mask):
    from circuit2spice.wires import build_node_map
    return build_node_map(img, dets, bridge_px=6, method="unet",
                          drop_outside="none", binary=mask)


if __name__ == "__main__":
    sys.exit(main())
