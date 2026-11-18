"""Batch-simulate a generated schematic to prove the .asc files really run."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit2spice.graph import CircuitGraph, Component, Detection
from circuit2spice.asc_writer import write_asc
from circuit2spice import ltspice


def comp(ref, cls, nodes, value, x, y, rot="R0"):
    return Component(ref=ref, cls=cls,
                     det=Detection(cls, x - 30, y - 30, x + 30, y + 30),
                     nodes=nodes, value=value, orientation=rot)


g = CircuitGraph(
    components=[
        comp("V1", "source_voltage", [1, 0], "AC 1", 100, 300),
        comp("R1", "resistor", [1, 2], "160", 250, 100, "R90"),
        comp("L1", "inductor", [2, 3], "1m", 420, 100, "R90"),
        comp("C1", "capacitor", [3, 0], "1u", 560, 300),
    ], n_nodes=4, image_size=(650, 420))

out = Path("outputs/smoke"); out.mkdir(parents=True, exist_ok=True)
asc = out / "rlc_ac.asc"
write_asc(g, asc, analysis=".ac dec 200 10 1Meg")
res = ltspice.simulate(asc)
log = res["log"]
print("raw file:", res["raw"])
for line in log.splitlines():
    if re.search(r"error|warning|fatal|cannot", line, re.I):
        print("  LOG:", line.strip())
print("log tail:", " | ".join(l.strip() for l in log.splitlines()[-4:] if l.strip()))
# Theoretical series-RLC resonance, for the report's cross-check.
import math
print(f"expected resonance f0 = {1/(2*math.pi*math.sqrt(1e-3*1e-6)):.1f} Hz")
print("SIM OK" if res["raw"] else "SIM FAILED")
