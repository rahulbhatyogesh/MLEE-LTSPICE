"""Run the pipeline on real CGHD photos using ground-truth boxes."""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit2spice.pipeline import process

root = Path("data/cghd/drafter_1")
stems = sys.argv[1:] or ["C1_D1_P1", "C2_D1_P1", "C3_D1_P1"]
out = Path("outputs/try"); out.mkdir(parents=True, exist_ok=True)

for stem in stems:
    img = root / "images" / f"{stem}.jpg"
    xml = root / "annotations" / f"{stem}.xml"
    if not img.exists():
        print(f"!! missing {img}"); continue
    r = process(img, gt_xml=xml, out_dir=out, verify=True, stem=stem)
    g = r.graph
    kinds = collections.Counter(c.cls for c in g.components)
    print(f"\n=== {stem} ===")
    print(f"  detections={len(r.detections)}  components={len(g.components)}  "
          f"wire-fragments->nodes={r.node_map.n_nodes}->{g.n_nodes}")
    print(f"  {dict(kinds)}")
    incomplete = [c.ref for c in g.components if not c.is_complete]
    print(f"  incomplete terminals: {incomplete or 'none'}")
    for w in g.warnings[:4]:
        print(f"  warn: {w}")
    if r.verification:
        print(f"  LTspice: {r.verification.summary()}")
