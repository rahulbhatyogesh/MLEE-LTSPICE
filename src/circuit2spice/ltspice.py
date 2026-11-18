"""Drive the real LTspice binary from Python.

LTspice has an undocumented-but-stable command line. Two flags matter here:

``-netlist <file.asc>``
    Extracts the netlist LTspice *itself* derives from the schematic and writes
    it next to the file as ``.net``. This is the ground truth for whether our
    generated geometry actually connects the way we intended.

``-b -Run <file.asc>``
    Batch-simulates and writes ``.raw`` plus a ``.log``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CANDIDATES = (
    Path("C:/Program Files/ADI/LTspice/LTspice.exe"),
    Path("C:/Program Files/LTC/LTspiceXVII/XVIIx64.exe"),
    Path("C:/Program Files (x86)/LTC/LTspiceIV/scad3.exe"),
)


def find_exe() -> Path | None:
    for c in CANDIDATES:
        if c.is_file():
            return c
    return None


def available() -> bool:
    return find_exe() is not None


class LTspiceError(RuntimeError):
    pass


def _run(args: list, timeout: int) -> subprocess.CompletedProcess:
    exe = find_exe()
    if exe is None:
        raise LTspiceError("LTspice is not installed")
    return subprocess.run(
        [str(exe), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def netlist(asc_path: str | Path, timeout: int = 90) -> str:
    """Return the netlist LTspice extracts from ``asc_path``."""
    asc = Path(asc_path).resolve()
    net = asc.with_suffix(".net")
    if net.exists():
        net.unlink()
    _run(["-netlist", str(asc)], timeout)
    if not net.exists():
        raise LTspiceError(f"LTspice produced no netlist for {asc.name}")
    return net.read_text(errors="ignore")


def simulate(asc_path: str | Path, timeout: int = 120) -> dict:
    """Batch-simulate and return ``{"log": ..., "raw": Path | None}``."""
    asc = Path(asc_path).resolve()
    _run(["-b", "-Run", str(asc)], timeout)
    log = asc.with_suffix(".log")
    raw = asc.with_suffix(".raw")
    return {
        "log": log.read_text(errors="ignore") if log.exists() else "",
        "raw": raw if raw.exists() else None,
    }


#: How many node tokens each SPICE element prefix carries in a netlist line.
_NODE_COUNT = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "Q": 3, "M": 4, "S": 4, "X": 3}


def parse_netlist(text: str) -> dict:
    """ref -> list of node names, for the element lines of a SPICE deck."""
    out: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        ref = parts[0]
        n = _NODE_COUNT.get(ref[0].upper())
        if n is None or len(parts) < 1 + n:
            continue
        out[ref] = parts[1 : 1 + n]
    return out
