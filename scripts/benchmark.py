"""End-to-end benchmark: photo -> netlist, scored against CGHD's reference schematics.

Drafter 1 has 12 reference LTspice files and 96 photographs (12 circuits x 2
drawings x 4 photos), so every photo has a ground-truth netlist.

Run with ground-truth boxes to measure the topology stage alone, or with
--weights to measure the full system.
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

from circuit2spice import ltspice
from circuit2spice.detect import detections_from_voc, nms_across_classes
from circuit2spice.netcompare import compare
from circuit2spice.pipeline import _load_and_scale
from circuit2spice.topology import build_circuit

ROOT = Path("data/cghd/drafter_1")
REF_SRC = ROOT / "spice"
REF_DST = Path("outputs/ref_netlists")


def normalise_asc(text: str) -> str:
    """Older LTspice allowed a free-text title line; 26.x aborts on it."""
    out, seen = [], False
    for line in text.splitlines():
        s = line.strip()
        if not seen and s and not s.startswith(("Version", "SHEET")):
            continue
        if s.startswith("SHEET"):
            seen = True
        out.append(line)
    return "\r\n".join(out) + "\r\n"


def load_references() -> dict:
    REF_DST.mkdir(parents=True, exist_ok=True)
    refs = {}
    for asc in sorted(REF_SRC.glob("*.asc")):
        local = REF_DST / asc.name
        cached = local.with_suffix(".net")
        if not cached.exists():
            local.write_text(
                normalise_asc(asc.read_text(errors="ignore", encoding="latin-1")),
                encoding="latin-1", newline="",
            )
            try:
                ltspice.netlist(local, timeout=90)
            except Exception as e:
                print(f"  !! {asc.stem}: {e}")
                continue
        refs[asc.stem] = ltspice.parse_netlist(cached.read_text(errors="ignore"))
    return refs


def predicted_elements(graph) -> dict:
    """{ref: [net names]}, with each unresolved port given a unique name so it
    can never accidentally match the reference."""
    out, nc = {}, 0
    for c in graph.components:
        nets = []
        for n in c.nodes:
            if n >= 0:
                nets.append(graph.node_name(n))
            else:
                nc += 1
                nets.append(f"NC_{nc}")
        out[c.ref] = nets
    return out


def run(bridge_px, detector=None, limit=None, method="flatfield", drop="mask", thin_q=50.0, reconnect_px=0.0):
    refs = load_references()
    xmls = sorted(ROOT.glob("annotations/*.xml"))
    if limit:
        xmls = xmls[:limit]
    rows = []
    for xml in xmls:
        circuit = xml.stem.split("_")[0]
        if circuit not in refs:
            continue
        img_p = ROOT / "images" / f"{xml.stem}.jpg"
        if not img_p.exists():
            continue
        if detector is None:
            dets = detections_from_voc(xml)
        else:
            dets = nms_across_classes(detector(str(img_p)))
        img, dets = _load_and_scale(img_p, dets, 1600)
        g, _ = build_circuit(img, dets, bridge_px=bridge_px, method=method,
                             drop_outside=drop, thin_q=thin_q,
                             reconnect_px=reconnect_px)
        has_gnd = any(d.cls == "gnd" for d in dets)
        rows.append(compare(xml.stem, circuit, predicted_elements(g),
                            refs[circuit], pin_ground=has_gnd))
    return rows


def summarise(rows, label):
    if not rows:
        print(f"{label}: no results"); return {}
    exact = sum(r.exact for r in rows) / len(rows)
    bom = statistics.mean(r.bom_f1 for r in rows)
    sig = statistics.mean(r.sig_f1 for r in rows)
    print(f"{label}: images={len(rows)}  exact-topology={exact:.3f}  "
          f"BoM-F1={bom:.3f}  signature-F1={sig:.3f}")
    return {"images": len(rows), "exact": round(exact, 4),
            "bom_f1": round(bom, 4), "sig_f1": round(sig, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="ablate bridge_px")
    ap.add_argument("--bridge", type=int, default=None)
    ap.add_argument("--weights", default=None, help="YOLO weights for the full system")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--method", default="flatfield")
    ap.add_argument("--binablate", action="store_true")
    ap.add_argument("--out", default="outputs/benchmark.json")
    args = ap.parse_args()

    detector = None
    if args.weights:
        from circuit2spice.detect import YoloDetector
        detector = YoloDetector(args.weights)

    results = {}
    if args.binablate:
        combos = [("adaptive","none",None), ("flatfield","none",None),
                  ("flatfield","mask",None), ("flatfield","mask",50.0),
                  ("flatfield","mask",60.0), ("flatfield","mask",70.0),
                  ("flatfield","mask",80.0)]
        for m, d, q in combos:
            rows = run(args.bridge, detector, args.limit, m, d, q)
            results[f"{m}|{d}|thin{q}"] = summarise(rows, f"{m:10} outside={d:5} thin_q={str(q):5}")
    elif args.sweep:
        for b in [0, 1, 2, 3, 4]:
            t0 = time.time()
            rows = run(b, detector, args.limit, args.method)
            results[f"bridge_{b}"] = summarise(rows, f"bridge_px={b}")
            results[f"bridge_{b}"]["secs"] = round(time.time() - t0, 1)
        best = max(results, key=lambda k: results[k]["exact"] + results[k]["sig_f1"])
        print(f"\nBEST: {best} -> {results[best]}")
    else:
        rows = run(args.bridge, detector, args.limit, args.method)
        tag = "full-system" if detector else "gt-boxes"
        results[tag] = summarise(rows, tag)
        per = collections.defaultdict(list)
        for r in rows:
            per[r.circuit].append(r)
        print("\nper circuit:")
        for c in sorted(per, key=lambda s: int(s[1:])):
            rs = per[c]
            print(f"  {c:4} n={len(rs):2} exact={sum(x.exact for x in rs)}/{len(rs)}"
                  f"  sig-F1={statistics.mean(x.sig_f1 for x in rs):.3f}"
                  f"  ({rs[0].n_pred} vs {rs[0].n_ref} elements)")
        results["rows"] = [r.as_row() for r in rows]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
