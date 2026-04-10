"""
gridder.py – 2D grid visualisation and proximity scoring for ELS matches.

Background
----------
When a linear sequence of letters is laid out in rows of width *w*, two letters
that are exactly *w* positions apart in the sequence land in the **same column**
on consecutive rows — a phenomenon exploited by Bible Code researchers to display
multiple ELS patterns converging on a single grid.

This module provides:

1. ``get_grid(compact_bytes, start_index, end_index, width)``
   Extract a contiguous segment of the Torah compact bytes and reshape it into
   a 2-D NumPy array of ``uint8`` values (1–22 = Hebrew letters, 0 = padding).

2. ``optimal_width_pair(m1, m2, ...)``
   Scan every integer width in ``[min_width, max_width]`` and return the width
   at which the **Manhattan distance** between the two match start positions is
   minimised.

3. ``cluster_score_at_width(matches, width)``
   Compute the mean pairwise Manhattan distance of all match start positions
   when the corpus is wrapped at *width*.  Lower = tighter cluster.

4. ``find_table(matches, ...)``
   Scan all widths in ``[min_width, max_width]`` and return a :class:`Table`
   with the width that minimises the cluster score.

5. :class:`Table` dataclass
   Stores the optimal *width*, the participating :class:`engine.Match` objects,
   and the *cluster_score* at that width.

Coordinate system
-----------------
All positions are in **global corpus letter indices** (i.e. ``Match.start``).
When the Torah is wrapped at width *w*:

    row = global_index // w
    col = global_index  % w

Manhattan distance between two positions ``p`` and ``q`` at width *w*:

    |p // w − q // w|  +  |p % w − q % w|
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from engine import Match

# Mirror engine._HEB_BASE so this module can decode letters without importing engine.
_HEB_BASE = 0x05D0


# ── Grid extraction ────────────────────────────────────────────────────────────

def get_grid(
    compact_bytes: bytes,
    start_index: int,
    end_index: int,
    width: int,
) -> np.ndarray:
    """
    Extract ``compact_bytes[start_index:end_index]`` and wrap it into a 2-D
    NumPy array of shape ``(rows, width)``.

    Parameters
    ----------
    compact_bytes :
        The 1-byte-per-letter compact Torah corpus, as returned by
        ``ELSEngine.corpus_bytes`` or ``ELSEngine.corpus_bytes_for()``.
    start_index : int
        First letter index to include (inclusive).
    end_index : int
        Last letter index to include (exclusive).
    width : int
        Number of columns in the output grid.  Must be ≥ 1.

    Returns
    -------
    np.ndarray  shape (rows, width), dtype uint8
        Values 1–22 correspond to Hebrew letters כ–ת (compact encoding);
        value 0 fills any padding cells in the final row.
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width!r}")
    start_index = max(0, start_index)
    end_index   = min(len(compact_bytes), end_index)

    segment = compact_bytes[start_index:end_index]
    n    = len(segment)
    rows = (n + width - 1) // width      # ceiling division
    pad  = rows * width - n              # zero-padding for last row
    arr  = np.frombuffer(segment + bytes(pad), dtype=np.uint8).reshape(rows, width)
    return arr.copy()                    # frombuffer returns read-only; copy makes it writable


def decode_grid(grid: np.ndarray) -> list[str]:
    """
    Convert a numeric grid (as returned by :func:`get_grid`) to a list of
    strings, one per row, with Hebrew Unicode letters and spaces for padding.

    Useful for quick terminal inspection::

        for row in decode_grid(get_grid(eng.corpus_bytes, 0, 200, 20)):
            print(row)
    """
    rows = []
    for row in grid:
        rows.append("".join(
            chr(int(b) + _HEB_BASE - 1) if 1 <= b <= 22 else " "
            for b in row
        ))
    return rows


# ── Proximity helpers ──────────────────────────────────────────────────────────

def _grid_pos(global_index: int, width: int) -> tuple[int, int]:
    """Return ``(row, col)`` for a global corpus letter index at the given width."""
    return divmod(global_index, width)


def _manhattan(pos1: int, pos2: int, width: int) -> int:
    """
    Manhattan distance between two global corpus positions when wrapped at *width*.

    This is an integer, so this hot-path helper avoids float arithmetic.
    """
    r1, c1 = divmod(pos1, width)
    r2, c2 = divmod(pos2, width)
    return abs(r1 - r2) + abs(c1 - c2)


def optimal_width_pair(
    m1: Match,
    m2: Match,
    min_width: int = 2,
    max_width: int = 2000,
) -> tuple[int, int]:
    """
    Find the grid width at which the Manhattan distance between *m1* and *m2*
    is minimised.

    Scans every integer width in ``[min_width, max_width]`` exhaustively.

    Parameters
    ----------
    m1, m2 :
        Two :class:`engine.Match` objects to compare.
    min_width, max_width :
        Search range for the width (default 2–2000).

    Returns
    -------
    tuple[int, int]
        ``(optimal_width, min_manhattan_distance)``
    """
    if min_width < 1:
        raise ValueError(f"min_width must be >= 1, got {min_width!r}")
    if max_width < min_width:
        raise ValueError(f"max_width ({max_width}) must be >= min_width ({min_width})")

    p1, p2 = m1.start, m2.start
    best_w = min_width
    best_d = _manhattan(p1, p2, min_width)

    for w in range(min_width + 1, max_width + 1):
        d = _manhattan(p1, p2, w)
        if d < best_d:
            best_d = d
            best_w = w
            if best_d == 0:
                break  # can't do better than zero

    return best_w, best_d


def cluster_score_at_width(matches: Sequence[Match], width: int) -> float:
    """
    Compute the mean pairwise Manhattan distance of all match start positions
    when the corpus is wrapped at *width*.

    A lower score means the matches cluster more tightly on the grid at this
    width — exactly what Bible Code researchers look for when choosing a display
    width for convergence tables.

    Parameters
    ----------
    matches :
        Sequence of :class:`engine.Match` objects (at least 2 required; returns
        ``0.0`` for fewer than 2).
    width : int
        Grid width to evaluate.

    Returns
    -------
    float
        Mean pairwise Manhattan distance.  ``0.0`` when fewer than 2 matches
        are provided.
    """
    positions = [m.start for m in matches]
    n = len(positions)
    if n < 2:
        return 0.0

    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _manhattan(positions[i], positions[j], width)
            count += 1
    return total / count


# ── Table dataclass ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Table:
    """
    A 2-D ELS convergence table.

    Attributes
    ----------
    width : int
        The grid width (number of columns) that minimises the cluster score
        for the participating matches.
    matches : tuple[Match, ...]
        The :class:`engine.Match` objects that converge at this width.
    cluster_score : float
        Mean pairwise Manhattan distance of all match start positions at
        *width*.  Lower = tighter spatial convergence.
    """
    width: int
    matches: tuple  # tuple[Match, ...] — generic annotation avoids engine import at class-def time
    cluster_score: float


# ── Table discovery ────────────────────────────────────────────────────────────

def find_table(
    matches: Sequence[Match],
    min_width: int = 2,
    max_width: int = 2000,
) -> Table:
    """
    Scan all grid widths in ``[min_width, max_width]`` and return a
    :class:`Table` at the width where the matches cluster most tightly.

    The search is exhaustive: for each candidate width *w*, the mean pairwise
    Manhattan distance of all match start positions is computed.  The width
    with the lowest score is selected.

    Parameters
    ----------
    matches :
        Sequence of :class:`engine.Match` objects to evaluate.  Typically the
        output of :meth:`ELSEngine.search` filtered to interesting hits.
    min_width, max_width :
        Width search range (default 2–2000).

    Returns
    -------
    Table
        The :class:`Table` with the optimal *width* and computed *cluster_score*.

    Notes
    -----
    Complexity is O((max_width − min_width) × n²) where *n* = ``len(matches)``.
    For typical researcher use (n < 20, range 2–2000) this is near-instant.
    """
    if len(matches) < 2:
        raise ValueError("find_table() requires at least 2 matches.")
    if min_width < 1:
        raise ValueError(f"min_width must be >= 1, got {min_width!r}")
    if max_width < min_width:
        raise ValueError(f"max_width ({max_width}) must be >= min_width ({min_width})")

    best_w     = min_width
    best_score = cluster_score_at_width(matches, min_width)

    for w in range(min_width + 1, max_width + 1):
        score = cluster_score_at_width(matches, w)
        if score < best_score:
            best_score = score
            best_w     = w

    return Table(
        width=best_w,
        matches=tuple(matches),
        cluster_score=best_score,
    )
