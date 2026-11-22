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

import struct
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


_RAW_MARKER = "Binary:\n".encode("utf-16-le")


def read_raw(raw_path: str | Path) -> dict:
    """Parse an LTspice binary ``.raw`` into ``{"plotname": str, "data": {name: [v]}}``.

    Two details of the format, both established by probing a real file rather
    than assumed, and both silently fatal if guessed wrong:

    - the header is **UTF-16LE**, terminated by the literal ``Binary:`` line;
    - in a ``real`` plot the **first** variable is stored as a float64 and
      every other variable as a float32. Reading all of them as doubles
      "succeeds" and returns numbers like 8.9e245.

    A ``complex`` plot (``.ac``) stores two float64 per value instead.

    One further quirk: in a transient plot LTspice sets the **sign bit** on
    roughly half the timestamps as an internal flag, so the raw doubles
    alternate in sign. The magnitude is the real time value, and taking the
    absolute value makes the axis monotonic again. Left alone, the time column
    runs negative and any plot drawn from it is scrambled -- which looks like a
    simulation fault rather than a parsing one.
    """
    data = Path(raw_path).read_bytes()
    i = data.find(_RAW_MARKER)
    if i < 0:
        raise LTspiceError(f"{raw_path}: not a binary LTspice raw file")
    head = data[:i].decode("utf-16-le")
    body = data[i + len(_RAW_MARKER):]

    meta = {}
    for line in head.splitlines():
        if ":" in line and not line.startswith("	"):
            k, _, v = line.partition(":")
            meta[k.strip().lower()] = v.strip()
    names = [ln.split("	")[2] for ln in head.splitlines() if ln.startswith("	")]
    n_pts = int(meta.get("no. points", "0"))
    complex_ = "complex" in meta.get("flags", "")

    out: dict = {n: [] for n in names}
    off = 0
    for _ in range(n_pts):
        for j, name in enumerate(names):
            if complex_:
                re_, im = struct.unpack_from("<dd", body, off); off += 16
                out[name].append(complex(re_, im))
            elif j == 0:
                out[name].append(struct.unpack_from("<d", body, off)[0]); off += 8
            else:
                out[name].append(struct.unpack_from("<f", body, off)[0]); off += 4
    # Undo the sign-bit flag LTspice sets on transient timestamps.
    if names and names[0].lower() == "time":
        out[names[0]] = [abs(v) for v in out[names[0]]]

    return {"plotname": meta.get("plotname", ""), "n_points": n_pts, "data": out}


def operating_point(raw_path: str | Path) -> dict:
    """``{name: value}`` for a single-point ``.op`` run."""
    r = read_raw(raw_path)
    return {k: v[0] for k, v in r["data"].items() if v}


#: How many node tokens each SPICE element prefix carries in a netlist line.
#:
#: ``X`` is pinned at 3 because the only subcircuit we place is the 3-pin ideal
#: op-amp. A wider one -- CGHD's reference C1 instantiates an 8-pin NE555 --
#: parses short, which is harmless only because such a part maps to
#: ``other_component`` and is never emitted or scored.
_NODE_COUNT = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2, "Q": 3, "M": 4, "S": 4, "X": 3}


def _ref(token: str) -> str:
    """The instance name, as *we* write it, from a netlist's first token.

    LTspice marks a symbol whose pins carry a ``SpiceOrder`` by writing the
    element letter, a section sign, and then the instance name::

        X§X1 N003 0 N004 opamp Aol=100K ;§pnba invin)noninvin)out

    so an op-amp we wrote as ``X1`` comes back as ``X§X1`` and compares equal
    to nothing. Every op-amp therefore failed the LTspice round-trip -- the
    one check in this project that is supposed to catch a schematic LTspice
    reads differently from how we meant it -- as "1 component absent, 1
    unexpected", on the strength of a name. Nothing else about the line is
    wrong: the nodes after the marker are exactly the ones we placed.
    """
    return token.split("§")[-1] if "§" in token else token


#: Node tokens each prefix carries *as LTspice writes them*, which is not the
#: same as the port count we model: LTspice emits a BJT's substrate node too,
#: so ``Q1 c b e sub MODEL`` has four nodes before the value.
_NETLIST_NODE_COUNT = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "D": 2,
                       "Q": 4, "M": 4, "S": 4}


def parse_netlist_values(text: str) -> dict:
    """``ref -> value text`` for the element lines of a SPICE deck.

    The value is everything after the nodes, so a multi-token value such as
    ``SINE(0 20m 400)`` survives intact.
    """
    out: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("*", ".")):
            continue
        parts = line.split()
        n = _NETLIST_NODE_COUNT.get(parts[0][0].upper())
        if n is None or len(parts) < 1 + n:
            continue
        out[_ref(parts[0])] = " ".join(parts[1 + n:]).strip()
    return out


def parse_netlist(text: str) -> dict:
    """ref -> list of node names, for the element lines of a SPICE deck."""
    out: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        ref = _ref(parts[0])
        n = _NODE_COUNT.get(parts[0][0].upper())
        if n is None or len(parts) < 1 + n:
            continue
        out[ref] = parts[1 : 1 + n]
    return out
