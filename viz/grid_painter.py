"""
viz/grid_painter.py — 2D ELS matrix renderer for CodeSearch-Ultra.

For a given verse start position and skip value, this module extracts the
surrounding ~1000 characters of the compact corpus and wraps them into a
2D matrix of width = |skip|.  When an ELS word is encoded at that skip, it
appears as a vertical (or diagonal) column in the matrix — revealing
"cross-word" convergences with neighbouring ELS patterns.

Public API
----------
    paint_grid(corpus_bytes, start, skip, word, *, context_chars, output_format)
        → GridResult dataframe (Polars) or rendered HTML string

    highlight_positions(grid_array, match_positions) → dict of cell highlights

Internal pipeline
-----------------
  1. Determine window: [max(0, start - context_chars//2) … start + context_chars//2]
  2. Slice corpus bytes in that window.
  3. Reshape numpy array to (rows, |skip|).
  4. Mark cells that belong to the primary ELS word.
  5. Scan every column and anti-diagonal for secondary ELS sub-sequences
     (cross-word detection).
  6. Return a structured GridResult.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

# ── backend path ──────────────────────────────────────────────────────────────
_BACKEND = Path(__file__).resolve().parent.parent / "backend"


_HEB_BASE = 0x05D0   # mirrors engine._HEB_BASE
_ENG_OFFSET = 27     # mirrors engine._ENG_COMPACT_OFFSET


def _decode_byte(b: int) -> str:
    """Decode one compact byte to its Unicode character."""
    if b == 0:
        return " "
    if b > _ENG_OFFSET:
        return chr(b - _ENG_OFFSET - 1 + 65)
    return chr(b + _HEB_BASE - 1)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class GridResult:
    """
    Full output of ``paint_grid``.

    Attributes
    ----------
    grid_array : np.ndarray  shape (rows, width)
        Raw compact-byte values.  0 = padding.
    width : int
        Number of columns (== |skip| clamped to ≥ 2).
    window_start : int
        Global corpus offset of the first byte in the grid.
    word_cells : list[tuple[int,int]]
        (row, col) positions of the primary ELS word letters.
    crossword_hits : list[CrosswordHit]
        Any secondary patterns detected by column/diagonal scan.
    html : str
        Colour-coded HTML table ready for embedding in the dashboard.
    dataframe : pl.DataFrame
        Flat cell export: row, col, letter, is_primary, is_crossword
    """

    grid_array: np.ndarray
    width: int
    window_start: int
    word_cells: list[tuple[int, int]]
    crossword_hits: list[CrosswordHit]
    satellite_hits: list[SatelliteHit]
    html: str
    dataframe: pl.DataFrame


@dataclass(frozen=True)
class CrosswordHit:
    """A secondary ELS sub-sequence detected in the grid."""
    direction: str         # "vertical" | "diagonal_fwd" | "diagonal_bwd"
    letters: str           # decoded unicode letters
    start_row: int
    start_col: int
    length: int


@dataclass(frozen=True)
class SatelliteHit:
    """A Tactical Vocabulary root found in an adjacent cell cluster."""
    root: str           # Hebrew 3-letter root (decoded)
    meaning: str        # English gloss
    cells: tuple[tuple[int, int], ...]  # (row, col) positions


# ── Tactical Vocabulary ──────────────────────────────────────────────────
TACTICAL_VOCABULARY: dict[str, str] = {
    "אש":  "Fire",
    "דם":  "Blood",
    "קץ":  "End",
    "מות": "Death",
    "חרב": "Sword",
    "מלכ": "King",
    "אל":  "God",
    "שלום": "Peace",
    "ברית": "Covenant",
    "עם":  "People",
    "ארץ": "Land",
    "משיח": "Messiah",
    "שמש": "Sun",
    "קול": "Voice",
    "יום": "Day",
    "עת":  "Time",
    "אור": "Light",
    "חשך": "Darkness",
}


# ── Core function ─────────────────────────────────────────────────────────────

def paint_grid(
    corpus_bytes: bytes,
    match_start: int,
    skip: int,
    word: str,
    *,
    context_chars: int = 1_000,
    output_format: Literal["result", "html", "dataframe"] = "result",
    min_crossword_len: int = 3,
    english_label: str = "",
    scan_satellites: bool = True,
) -> GridResult:
    """
    Render a 2D ELS matrix for a single match.

    Parameters
    ----------
    corpus_bytes : bytes
        Compiled compact corpus (from ``UltraSearchEngine.corpus_bytes()``).
    match_start : int
        Global letter index of the first character of the primary ELS match.
    skip : int
        ELS skip distance.  The grid width is set to ``max(2, abs(skip))``.
    word : str
        Hebrew Unicode word (used only for cell annotation / HTML label).
    context_chars : int
        Number of corpus characters to include in the window (default 1000).
    output_format : {"result", "html", "dataframe"}
        Controls whether the full GridResult, only the HTML string, or only the
        Polars DataFrame is returned.  When "html" or "dataframe", the function
        still returns a full GridResult but this flag is preserved for callers
        that only need one piece.
    min_crossword_len : int
        Minimum letter count for a secondary ELS run to be reported as a
        CrosswordHit (default 3).

    Returns
    -------
    GridResult
    """
    abs_skip = max(2, abs(skip))
    half     = context_chars // 2

    window_start = max(0, match_start - half)
    window_end   = min(len(corpus_bytes), window_start + context_chars)
    window_end   = max(window_end, window_start + abs_skip)  # at least one row

    segment = corpus_bytes[window_start:window_end]
    n_bytes = len(segment)

    rows  = (n_bytes + abs_skip - 1) // abs_skip
    pad   = rows * abs_skip - n_bytes
    arr   = np.frombuffer(segment + bytes(pad), dtype=np.uint8).reshape(rows, abs_skip).copy()

    # ── Mark primary word cells ───────────────────────────────────────────────
    word_cells: list[tuple[int, int]] = []
    local_start = match_start - window_start
    if local_start >= 0:
        for i in range(len(word)):
            pos = local_start + i * abs(skip)
            if pos < rows * abs_skip:
                r, c = divmod(pos, abs_skip)
                word_cells.append((r, c))

    # ── Detect cross-words (column scan + diagonal scan) ─────────────────────
    crossword_hits = _scan_crosswords(arr, min_len=min_crossword_len)

    # ── Build Polars DataFrame ────────────────────────────────────────────────
    primary_set   = set(word_cells)
    crossword_set = {(h.start_row + i, h.start_col)
                     for h in crossword_hits
                     for i in range(h.length)
                     if h.direction == "vertical"}

    cell_rows: list[dict] = []
    for r in range(rows):
        for c in range(abs_skip):
            b = int(arr[r, c])
            cell_rows.append({
                "row":          r,
                "col":          c,
                "byte":         b,
                "letter":       _decode_byte(b),
                "is_primary":   (r, c) in primary_set,
                "is_crossword": (r, c) in crossword_set,
            })

    df = pl.DataFrame(cell_rows)

    # ── Satellite Term Detection ───────────────────────────────────────────────
    satellite_hits: list[SatelliteHit] = []
    satellite_set: set[tuple[int, int]] = set()
    if scan_satellites and word_cells:
        satellite_hits = _scan_satellites(arr, word_cells)
        for sh in satellite_hits:
            satellite_set.update(sh.cells)

    # ── Render HTML ──────────────────────────────────────────────────────
    html = _render_html(arr, primary_set, crossword_set, satellite_set, abs_skip, word, english_label)

    return GridResult(
        grid_array=arr,
        width=abs_skip,
        window_start=window_start,
        word_cells=word_cells,
        crossword_hits=crossword_hits,
        satellite_hits=satellite_hits,
        html=html,
        dataframe=df,
    )


# ── Satellite Scanner ─────────────────────────────────────────────────────

def _scan_satellites(
    arr: np.ndarray,
    word_cells: list[tuple[int, int]],
) -> list[SatelliteHit]:
    """
    Scan the 8-neighbourhood of every primary word cell for sequences
    that match a root in TACTICAL_VOCABULARY.

    For each root, we try to spell it out in all 8 compass directions starting
    from each immediate neighbour of each primary cell.  A hit is recorded once
    per unique (root, first-cell) combination.
    """
    rows, cols = arr.shape
    hits: list[SatelliteHit] = []
    seen: set[tuple[str, tuple[int, int]]] = set()

    # 8 compass directions: (dr, dc)
    _DIRS = [
        (-1,  0), (1,  0), (0, -1), (0,  1),
        (-1, -1), (-1,  1), (1, -1), (1,  1),
    ]

    # Pre-encode each root to compact bytes for fast comparison
    _vocab_bytes: dict[str, bytes] = {}
    for root in TACTICAL_VOCABULARY:
        enc = bytes(_HEB_BASE + ord(ch) - 0x05D0 + 1 if ch >= '\u05D0' else 0 for ch in root)
        _vocab_bytes[root] = enc

    def _try_root(start_r: int, start_c: int) -> None:
        for root, gloss in TACTICAL_VOCABULARY.items():
            rlen = len(root)
            for dr, dc in _DIRS:
                cells: list[tuple[int, int]] = []
                match = True
                for k in range(rlen):
                    nr, nc = start_r + k * dr, start_c + k * dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        match = False
                        break
                    expected_byte = ord(root[k]) - _HEB_BASE + 1
                    if int(arr[nr, nc]) != expected_byte:
                        match = False
                        break
                    cells.append((nr, nc))
                if match and cells:
                    key = (root, cells[0])
                    if key not in seen:
                        seen.add(key)
                        hits.append(SatelliteHit(
                            root=root,
                            meaning=gloss,
                            cells=tuple(cells),
                        ))

    # Collect unique neighbours of all primary cells
    neighbour_seeds: set[tuple[int, int]] = set()
    for pr, pc in word_cells:
        for dr, dc in _DIRS:
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbour_seeds.add((nr, nc))

    for r, c in neighbour_seeds:
        _try_root(r, c)

    return hits


# ── Cross-word scanner ────────────────────────────────────────────────────────

def _scan_crosswords(arr: np.ndarray, *, min_len: int) -> list[CrosswordHit]:
    """
    Scan all columns and both main-diagonal directions for runs of non-zero bytes.

    A "run" is any contiguous sequence of *min_len* or more non-zero bytes
    in a given direction.
    """
    rows, cols = arr.shape
    hits: list[CrosswordHit] = []

    # Vertical scan (same column, consecutive rows)
    for c in range(cols):
        col_bytes = arr[:, c]
        _extract_runs(col_bytes, direction="vertical",
                      start_col=c, hits=hits, min_len=min_len)

    # Forward diagonal (row+1, col+1)
    for start_r in range(rows):
        diag_bytes = np.array([arr[start_r + i, i]
                                for i in range(min(rows - start_r, cols))])
        _extract_runs(diag_bytes, direction="diagonal_fwd",
                      start_col=0, start_row_offset=start_r,
                      hits=hits, min_len=min_len)

    # Backward diagonal (row+1, col-1)
    for start_c in range(cols):
        diag_bytes = np.array([arr[i, start_c - i]
                                for i in range(min(rows, start_c + 1))])
        _extract_runs(diag_bytes, direction="diagonal_bwd",
                      start_col=start_c, hits=hits, min_len=min_len)

    return hits


def _extract_runs(
    seq: np.ndarray,
    *,
    direction: str,
    start_col: int,
    start_row_offset: int = 0,
    hits: list[CrosswordHit],
    min_len: int,
) -> None:
    """Extract non-zero runs of length ≥ min_len from a 1-D byte array."""
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] == 0:
            i += 1
            continue
        j = i
        while j < n and seq[j] != 0:
            j += 1
        run_len = j - i
        if run_len >= min_len:
            letters = "".join(_decode_byte(int(b)) for b in seq[i:j])
            hits.append(CrosswordHit(
                direction=direction,
                letters=letters,
                start_row=start_row_offset + i,
                start_col=start_col,
                length=run_len,
            ))
        i = j


# ── HTML renderer ─────────────────────────────────────────────────────────────

_PRIMARY_COLOR    = "#FF6B35"   # orange — primary ELS word
_CROSSWORD_COLOR  = "#4ECDC4"   # teal   — cross-word hits
_SATELLITE_COLOR  = "#F7DC6F"   # amber  — satellite tactical vocabulary
_OVERLAP_COLOR    = "#9B59B6"   # purple — both primary + crossword
_DEFAULT_BG       = "#1A1A2E"   # dark background
_DEFAULT_FG       = "#E0E0E0"   # light foreground


def _render_html(
    arr: np.ndarray,
    primary_set: set[tuple[int, int]],
    crossword_set: set[tuple[int, int]],
    satellite_set: set[tuple[int, int]],
    width: int,
    word: str,
    english_label: str = "",
) -> str:
    """Return a self-contained HTML table representing the grid."""
    rows = arr.shape[0]

    td_base = (
        "padding:2px 3px;font-family:monospace;font-size:12px;"
        "border:1px solid #333;text-align:center;min-width:18px;"
    )

    cells_html: list[str] = []
    for r in range(rows):
        cells_html.append("<tr>")
        for c in range(width):
            b = int(arr[r, c])
            letter = _decode_byte(b) if b else "&nbsp;"
            is_p = (r, c) in primary_set
            is_x = (r, c) in crossword_set
            is_s = (r, c) in satellite_set

            if is_p and is_x:
                bg = _OVERLAP_COLOR
            elif is_p:
                bg = _PRIMARY_COLOR
            elif is_x:
                bg = _CROSSWORD_COLOR
            elif is_s:
                bg = _SATELLITE_COLOR
            else:
                bg = _DEFAULT_BG

            fg = "#1A1A2E" if is_s else ("#FFFFFF" if (is_p or is_x) else _DEFAULT_FG)
            title_attr = f' title="{english_label}"' if (is_p and english_label) else ""
            cells_html.append(
                f'<td style="{td_base}background:{bg};color:{fg};"{title_attr}>'  
                f"{letter}</td>"
            )
        cells_html.append("</tr>")

    table_style = (
        "border-collapse:collapse;direction:rtl;"
        f"background:{_DEFAULT_BG};color:{_DEFAULT_FG};"
    )
    label_suffix = f" — {english_label}" if english_label else ""
    legend = (
        "\u25a0 <span style='color:#FF6B35'>Primary</span> "
        "\u25a0 <span style='color:#4ECDC4'>Crossword</span> "
        "\u25a0 <span style='color:#F7DC6F'>Satellite</span> "
        "\u25a0 <span style='color:#9B59B6'>Overlap</span>"
    )
    header = (
        f'<caption style="color:#AAA;font-family:monospace;font-size:11px;">'
        f"ELS Grid \u2014 word={word}{label_suffix} width={width}<br>{legend}</caption>"
    )
    return (
        f'<table style="{table_style}">'
        f"{header}"
        f"{''.join(cells_html)}"
        f"</table>"
    )
