"""
reporter.py – Export ELS search results to CSV, Parquet, or colour-coded HTML grids.

Public API
----------
    export_results(matches, format='csv', filename='results') -> Path
    render_grid_to_html(grid, matches, grid_start_index, filename='grid') -> Path

Supported export formats
------------------------
    csv     – UTF-8 CSV, readable in Excel / LibreOffice without extra tools.
    parquet – Apache Parquet (columnar, compact, round-trips perfectly with Polars).

HTML grid colour scheme
-----------------------
    Each unique search word receives a distinct highlight colour (up to 8 words
    are pre-assigned; beyond that colours cycle).  Letter positions that are hit
    by more than one word are shown in an overlap colour (purple by default).
    Unhighlighted positions show as plain text on a white background.

Columns written (CSV / Parquet)
--------------------------------
    word            – search term (Hebrew Unicode)
    book            – book name of the first matched letter
    verse           – verse reference of the first matched letter
    skip            – ELS skip distance (positive = L→R, negative = R→L)
    start           – letter index of the first matched letter in the full corpus
    length          – number of letters in the match
    sequence        – actual Hebrew letters decoded from the corpus
    hebert_score    – cosine similarity vs. surrounding context (0.0 if not run)
    z_score         – corpus-level Z: (real_hits − μ_shuffle) / σ_shuffle
    is_significant  – True when z_score > threshold used in this run
    semantic_z_score – per-match semantic Z: (hebert_score − μ_null) / σ_null
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from engine import Match


# Column order used in every export
_COLUMNS = [
    "word",
    "book",
    "verse",
    "skip",
    "start",
    "length",
    "sequence",
    "hebert_score",
    "z_score",
    "is_significant",
    "semantic_z_score",
]


def _to_dataframe(matches: list[Match]) -> pl.DataFrame:
    """Convert a list of Match objects to a Polars DataFrame."""
    if not matches:
        return pl.DataFrame(schema={
            "word":             pl.Utf8,
            "book":             pl.Utf8,
            "verse":            pl.Utf8,
            "skip":             pl.Int32,
            "start":            pl.UInt32,
            "length":           pl.UInt32,
            "sequence":         pl.Utf8,
            "hebert_score":     pl.Float32,
            "z_score":          pl.Float32,
            "is_significant":   pl.Boolean,
            "semantic_z_score": pl.Float32,
        })

    return pl.DataFrame(
        {
            "word":             [m.word            for m in matches],
            "book":             [m.book            for m in matches],
            "verse":            [m.verse           for m in matches],
            "skip":             [m.skip            for m in matches],
            "start":            [m.start           for m in matches],
            "length":           [m.length          for m in matches],
            "sequence":         [m.sequence        for m in matches],
            "hebert_score":     [float(m.hebert_score)     for m in matches],
            "z_score":          [float(m.z_score)          for m in matches],
            "is_significant":   [m.is_significant          for m in matches],
            "semantic_z_score": [float(m.semantic_z_score) for m in matches],
        },
        schema={
            "word":             pl.Utf8,
            "book":             pl.Utf8,
            "verse":            pl.Utf8,
            "skip":             pl.Int32,
            "start":            pl.UInt32,
            "length":           pl.UInt32,
            "sequence":         pl.Utf8,
            "hebert_score":     pl.Float32,
            "z_score":          pl.Float32,
            "is_significant":   pl.Boolean,
            "semantic_z_score": pl.Float32,
        },
    ).select(_COLUMNS)


def export_results(
    matches: list[Match],
    format: Literal["csv", "parquet"] = "csv",
    filename: str = "results",
) -> Path:
    """
    Export *matches* to a file and return the resolved ``Path``.

    Parameters
    ----------
    matches : list[Match]
        Results from ``ELSEngine.search()`` (optionally enriched by
        ``stats.apply_significance()`` and ``validator`` scoring).
    format : {"csv", "parquet"}
        Output file format.  ``"csv"`` produces a UTF-8 CSV file; ``"parquet"``
        produces an Apache Parquet file.  Default: ``"csv"``.
    filename : str
        Base filename without extension.  The appropriate extension is appended
        automatically.  Relative paths are resolved against the current working
        directory.  Default: ``"results"``.

    Returns
    -------
    Path
        Absolute path to the file that was written.

    Raises
    ------
    ValueError
        If *format* is not ``"csv"`` or ``"parquet"``.
    """
    if format not in ("csv", "parquet"):
        raise ValueError(f"Unsupported export format: {format!r}. Choose 'csv' or 'parquet'.")

    df = _to_dataframe(matches)
    ext = ".csv" if format == "csv" else ".parquet"
    path = Path(filename).with_suffix(ext).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        df.write_csv(path)
    else:
        df.write_parquet(path)

    return path


# ── HTML grid rendering ────────────────────────────────────────────────────────

# Compact encoding base — mirrors engine._HEB_BASE
_HEB_BASE = 0x05D0

# Per-word highlight colours (background, dark-text foreground).  Up to 8 words
# get distinct colours; beyond that the palette cycles.
_WORD_COLOURS: list[tuple[str, str]] = [
    ("#3b82f6", "#ffffff"),  # blue
    ("#ef4444", "#ffffff"),  # red
    ("#22c55e", "#ffffff"),  # green
    ("#f59e0b", "#000000"),  # amber
    ("#8b5cf6", "#ffffff"),  # violet
    ("#06b6d4", "#000000"),  # cyan
    ("#f97316", "#ffffff"),  # orange
    ("#84cc16", "#000000"),  # lime
]
_OVERLAP_BG   = "#9333ea"   # purple — two or more words share this cell
_OVERLAP_FG   = "#ffffff"
_PLAIN_BG     = "#ffffff"
_PLAIN_FG     = "#1f2937"


def _build_highlight_map(
    matches: list[Match],
    grid_start_index: int,
    grid_end_index: int,
) -> dict[int, int]:
    """
    Return a mapping of ``global_index → word_index`` for every match letter
    that falls within ``[grid_start_index, grid_end_index)``.

    ``word_index`` is the position of the match's word in the unique-word list
    derived from the matches, so colour assignment is stable across calls.
    A special sentinel value of ``-1`` marks positions claimed by more than one
    word (overlap).

    Parameters
    ----------
    matches :
        ELS match objects (``Match.start``, ``Match.skip``, ``Match.length``,
        ``Match.word``).
    grid_start_index, grid_end_index :
        The global corpus range covered by the grid.

    Returns
    -------
    dict[int, int]
        ``{global_index: word_idx}`` where ``word_idx`` is 0-based per unique
        word, or ``-1`` for overlapping positions.
    """
    # Stable order of unique words (preserves first-seen order)
    seen: dict[str, int] = {}
    for m in matches:
        if m.word not in seen:
            seen[m.word] = len(seen)

    highlight: dict[int, int] = {}
    for m in matches:
        widx = seen[m.word]
        for j in range(m.length):
            pos = m.start + j * m.skip
            if grid_start_index <= pos < grid_end_index:
                if pos in highlight and highlight[pos] != widx:
                    highlight[pos] = -1   # overlap
                else:
                    highlight[pos] = widx
    return highlight


def render_grid_to_html(
    grid: "np.ndarray",
    matches: list[Match],
    grid_start_index: int,
    filename: str = "grid",
) -> Path:
    """
    Render a 2-D ELS grid as a colour-coded standalone HTML file.

    Each cell of the grid displays one Hebrew letter.  Letters that belong to
    an ELS match are highlighted with a word-specific background colour.
    Positions where two or more different words overlap are highlighted in
    purple.

    Parameters
    ----------
    grid : np.ndarray  shape (rows, width), dtype uint8
        As returned by :func:`gridder.get_grid`.  Values 1–22 = Hebrew letters
        (compact encoding); 0 = padding (shown as a non-breaking space).
    matches : list[Match]
        Match objects used to determine which cells to highlight.  The grid's
        coordinate range must encompass ``Match.start`` positions; positions
        outside ``[grid_start_index, grid_start_index + grid.size)`` are silently
        ignored.
    grid_start_index : int
        Global corpus letter index of the first cell in *grid* (i.e. the
        *start_index* that was passed to :func:`gridder.get_grid`).
    filename : str
        Base filename without extension; ``.html`` is appended automatically.
        Relative paths are resolved against the current working directory.

    Returns
    -------
    Path
        Absolute path of the HTML file written.
    """
    rows, width = grid.shape
    grid_end_index = grid_start_index + rows * width

    # Build word-colour legend
    unique_words: list[str] = []
    for m in matches:
        if m.word not in unique_words:
            unique_words.append(m.word)

    highlight_map = _build_highlight_map(matches, grid_start_index, grid_end_index)

    # ── CSS ───────────────────────────────────────────────────────────────────
    css_rules = [
        "body { font-family: 'Segoe UI', sans-serif; background: #f8fafc; margin: 2rem; direction: ltr; }",
        "h1 { font-size: 1.25rem; color: #1e293b; margin-bottom: 0.5rem; }",
        ".legend { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }",
        ".legend-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; }",
        ".swatch { width: 1rem; height: 1rem; border-radius: 3px; display: inline-block; }",
        "table { border-collapse: collapse; direction: ltr; }",
        "td { width: 1.8rem; height: 1.8rem; text-align: center; vertical-align: middle; "
        "     font-size: 1rem; border: 1px solid #e2e8f0; font-family: 'SBL Hebrew', 'Ezra SIL', serif; }",
        ".plain    { background: " + _PLAIN_BG + "; color: " + _PLAIN_FG + "; }",
        ".overlap  { background: " + _OVERLAP_BG + "; color: " + _OVERLAP_FG + "; font-weight: bold; }",
    ]
    for i, (bg, fg) in enumerate(_WORD_COLOURS):
        css_rules.append(f".word{i} {{ background: {bg}; color: {fg}; font-weight: bold; }}")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items: list[str] = []
    for i, word in enumerate(unique_words):
        ci = i % len(_WORD_COLOURS)
        bg, _ = _WORD_COLOURS[ci]
        legend_items.append(
            f'<div class="legend-item">'
            f'<span class="swatch" style="background:{bg}"></span>'
            f'<span>{html.escape(word)}</span>'
            f'</div>'
        )
    legend_items.append(
        f'<div class="legend-item">'
        f'<span class="swatch" style="background:{_OVERLAP_BG}"></span>'
        f'<span>Overlap</span>'
        f'</div>'
    )

    # ── Table rows ────────────────────────────────────────────────────────────
    table_rows: list[str] = []
    for r in range(rows):
        cells: list[str] = []
        for c in range(width):
            global_idx = grid_start_index + r * width + c
            b = int(grid[r, c])
            letter = chr(b + _HEB_BASE - 1) if 1 <= b <= 22 else "\u00a0"  # NBSP for padding
            letter_esc = html.escape(letter)

            if global_idx in highlight_map:
                widx = highlight_map[global_idx]
                if widx == -1:
                    css_cls = "overlap"
                else:
                    css_cls = f"word{widx % len(_WORD_COLOURS)}"
            else:
                css_cls = "plain"

            cells.append(f'<td class="{css_cls}" title="pos {global_idx}">{letter_esc}</td>')
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    # ── Full HTML document ────────────────────────────────────────────────────
    html_doc = (
        "<!DOCTYPE html>\n"
        '<html lang="he">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f'<title>ELS Grid — width {width}</title>\n'
        "<style>\n"
        + "\n".join(css_rules)
        + "\n</style>\n"
        "</head>\n"
        "<body>\n"
        f'<h1>ELS Grid — width {width} &nbsp;|&nbsp; '
        f'positions {grid_start_index}–{grid_end_index - 1}</h1>\n'
        '<div class="legend">\n'
        + "\n".join(legend_items)
        + "\n</div>\n"
        "<table>\n"
        + "\n".join(table_rows)
        + "\n</table>\n"
        "</body>\n"
        "</html>\n"
    )

    path = Path(filename).with_suffix(".html").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")
    return path
