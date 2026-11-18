"""LTspice symbol geometry.

Placing a symbol in a ``.asc`` file is only half the job: to wire it up you need
to know where its *pins* sit relative to the symbol origin, and how those pins
move when the symbol is rotated. Both are easy to get subtly wrong -- a
resistor's second pin is at y=96, not the y=80 you might assume from the drawn
body -- and the failure mode is silent: LTspice opens the schematic happily and
simply treats the component as unconnected.

So we read the pin coordinates out of LTspice's own ``.asy`` symbol files when
LTspice is installed, and fall back to a table that was captured from those same
files (LTspice 26.1.1) when it is not.

The rotation convention was verified empirically by netlisting a rotated
resistor with ``LTspice.exe -netlist``:

    R0 (x, y)    R90 (-y, x)    R180 (-x, -y)    R270 (y, -x)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ORIENTATIONS = ("R0", "R90", "R180", "R270")


@dataclass(frozen=True)
class Pin:
    name: str
    x: int
    y: int


@dataclass(frozen=True)
class SymbolDef:
    """Pin geometry for one LTspice symbol, in symbol-local coordinates."""

    name: str
    pins: tuple[Pin, ...]

    def pin_offsets(self, orientation: str = "R0") -> list[tuple[int, int]]:
        return [rotate(p.x, p.y, orientation) for p in self.pins]

    def pin_positions(
        self, origin: tuple[int, int], orientation: str = "R0"
    ) -> list[tuple[int, int]]:
        """Absolute sheet coordinates of each pin."""
        ox, oy = origin
        return [(ox + dx, oy + dy) for dx, dy in self.pin_offsets(orientation)]


def rotate(x: int, y: int, orientation: str) -> tuple[int, int]:
    """Apply an LTspice rotation to a symbol-local offset."""
    if orientation == "R0":
        return x, y
    if orientation == "R90":
        return -y, x
    if orientation == "R180":
        return -x, -y
    if orientation == "R270":
        return y, -x
    raise ValueError(f"unsupported orientation {orientation!r}")


# --------------------------------------------------------------------------
# Fallback geometry, captured from LTspice 26.1.1 lib/sym/*.asy
# --------------------------------------------------------------------------

_FALLBACK: dict[str, tuple[tuple[str, int, int], ...]] = {
    "res":              (("A", 16, 16), ("B", 16, 96)),
    "cap":              (("A", 16, 0), ("B", 16, 64)),
    "ind":              (("A", 16, 16), ("B", 16, 96)),
    "diode":            (("+", 16, 0), ("-", 16, 64)),
    "voltage":          (("+", 0, 16), ("-", 0, 96)),
    "current":          (("+", 0, 0), ("-", 0, 80)),
    "sw":               (("A", 0, 16), ("B", 0, 96)),
    "npn":              (("C", 64, 0), ("B", 0, 48), ("E", 64, 96)),
    "nmos":             (("D", 48, 0), ("G", 0, 80), ("S", 48, 96)),
    # 3-pin ideal op-amp: no supply rails to leave dangling, which matters
    # because a hand-drawn op-amp triangle almost never shows them.
    "OpAmps\\opamp":    (("invin", -32, 48), ("noninvin", -32, 80), ("out", 32, 64)),
}

#: Symbols that need a model library pulled in via a ``.lib`` directive.
REQUIRED_LIBS: dict[str, str] = {"OpAmps\\opamp": "opamp.sub"}


def ltspice_sym_dirs() -> list[Path]:
    """Candidate locations of the LTspice symbol library."""
    cands = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "LTspice" / "lib" / "sym",
        Path(os.environ.get("USERPROFILE", "")) / "Documents" / "LTspiceXVII" / "lib" / "sym",
        Path("C:/Program Files/ADI/LTspice/lib/sym"),
        Path("C:/Program Files/LTC/LTspiceXVII/lib/sym"),
    ]
    return [p for p in cands if p.is_dir()]


def _parse_asy(path: Path) -> tuple[Pin, ...]:
    pins: list[Pin] = []
    pending: tuple[int, int] | None = None
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("PIN "):
            parts = line.split()
            if len(parts) >= 3:
                pending = (int(parts[1]), int(parts[2]))
        elif line.startswith("PINATTR PinName") and pending is not None:
            pins.append(Pin(line.split(None, 2)[2].strip(), pending[0], pending[1]))
            pending = None
    return tuple(pins)


@lru_cache(maxsize=None)
def load_symbol(name: str) -> SymbolDef:
    """Look up a symbol by its LTspice name (e.g. ``res``, ``OpAmps\\opamp``).

    Prefers the installed ``.asy`` file; falls back to the captured table.
    """
    rel = name.replace("\\", "/")
    for sym_dir in ltspice_sym_dirs():
        asy = sym_dir / f"{rel}.asy"
        if asy.is_file():
            pins = _parse_asy(asy)
            if pins:
                return SymbolDef(name, pins)
    if name in _FALLBACK:
        return SymbolDef(name, tuple(Pin(*p) for p in _FALLBACK[name]))
    raise KeyError(f"unknown LTspice symbol {name!r}")


def available() -> bool:
    """True when a real LTspice symbol library was found."""
    return bool(ltspice_sym_dirs())
