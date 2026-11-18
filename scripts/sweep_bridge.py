"""Ablate the wire-bridging radius on real CGHD photos (ground-truth boxes)."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
from circuit2spice.detect import detections_from_voc
from circuit2spice.topology import build_circuit
from circuit2spice.evaluate import TopologyMetrics, score_graph
from circuit2spice.pipeline import _load_and_scale

root = Path("data/cghd/drafter_1")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
xmls = sorted(root.glob("annotations/*.xml"))[:n]

rows = []
for bridge in [0, 1, 2, 3, 4, 6]:
    acc, t0 = TopologyMetrics(), time.time()
    for xml in xmls:
        img_p = root / "images" / f"{xml.stem}.jpg"
        if not img_p.exists():
            continue
        dets = detections_from_voc(xml)
        img, dets = _load_and_scale(img_p, dets, 1600)
        g, _ = build_circuit(img, dets, bridge_px=bridge)
        score_graph(g, acc)
    row = {"bridge_px": bridge, **acc.as_row(), "secs": round(time.time() - t0, 1)}
    rows.append(row)
    print(f"bridge={bridge}: completeness={row['completeness']:.3f} "
          f"short_rate={row['short_rate']:.3f} connected={row['connected_frac']:.3f} "
          f"score={row['score']:.3f}  ({row['secs']}s, {row['components']} comps)")

best = max(rows, key=lambda r: r["score"])
print(f"\nBEST bridge_px={best['bridge_px']} (score {best['score']:.3f})")
Path("outputs").mkdir(exist_ok=True)
Path("outputs/bridge_sweep.json").write_text(json.dumps(rows, indent=2))
