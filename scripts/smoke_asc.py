"""Build known circuits as graphs and check LTspice agrees with the .asc we emit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from circuit2spice.graph import CircuitGraph, Component, Detection
from circuit2spice.netlist import to_netlist
from circuit2spice.verify import verify_graph


def comp(ref, cls, nodes, value, x, y, rot="R0"):
    return Component(ref=ref, cls=cls,
                     det=Detection(cls, x - 30, y - 30, x + 30, y + 30),
                     nodes=nodes, value=value, orientation=rot)


CASES = {
    "divider": CircuitGraph(
        components=[
            comp("V1", "source_voltage", [1, 0], "5", 100, 200),
            comp("R1", "resistor", [1, 2], "1k", 300, 120),
            comp("R2", "resistor", [2, 0], "2k", 300, 280),
        ], n_nodes=3, image_size=(400, 400)),
    "rc_lowpass": CircuitGraph(
        components=[
            comp("V1", "source_voltage", [1, 0], "1", 100, 200),
            comp("R1", "resistor", [1, 2], "1k", 300, 120, "R90"),
            comp("C1", "capacitor", [2, 0], "1u", 420, 280),
        ], n_nodes=3, image_size=(500, 400)),
    "series_rlc": CircuitGraph(
        components=[
            comp("V1", "source_voltage", [1, 0], "1", 100, 300),
            comp("R1", "resistor", [1, 2], "10", 250, 100, "R90"),
            comp("L1", "inductor", [2, 3], "1m", 420, 100, "R90"),
            comp("C1", "capacitor", [3, 0], "1u", 560, 300),
        ], n_nodes=4, image_size=(650, 420)),
}

out = Path(__file__).resolve().parents[1] / "outputs" / "smoke"
out.mkdir(parents=True, exist_ok=True)

fails = 0
for name, g in CASES.items():
    res = verify_graph(g, out / f"{name}.asc")
    print(f"--- {name}: {res.summary()}")
    if not res.ok:
        fails += 1
        for m in res.mismatches[:6]:
            print("    mismatch:", m)
        for f in res.floating_pins[:6]:
            print("    floating:", f)
        print("    LTspice saw:")
        for line in res.netlist_text.splitlines():
            if line.strip() and not line.startswith("*"):
                print("      ", line)
    (out / f"{name}.cir").write_text(to_netlist(g, title=name))

print(f"\n{len(CASES) - fails}/{len(CASES)} circuits verified")
sys.exit(1 if fails else 0)
