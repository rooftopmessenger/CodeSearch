"""
fetch_kjv_nt.py – Download the full KJV New Testament from the
scrollmapper/bible_databases project (public domain) and write one
plain-text file per NT book into the texts/ folder.

Output format (same as the existing stub files and the Koren Torah format):
    book_num  chapter  verse  word1 word2 ...

Usage:
    uv run fetch_kjv_nt.py
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

# URL: scrollmapper KJV CSV (public domain, MIT-licensed repository)
# Correct path confirmed via GitHub API: formats/csv/KJV.csv
# Columns: b (book 1-66), c (chapter), v (verse), t (text)
_KJV_CSV_URL = (
    "https://raw.githubusercontent.com/scrollmapper/"
    "bible_databases/master/formats/csv/KJV.csv"
)

# NT book canonical name (as it appears in KJV.csv Book column) → (filename, book_num)
# Note: the CSV uses Roman numerals (I/II/III) and "Revelation of John"
_NT_BOOKS: dict[str, tuple[str, int]] = {
    "Matthew":           ("40-Matthew.txt",        40),
    "Mark":              ("41-Mark.txt",            41),
    "Luke":              ("42-Luke.txt",            42),
    "John":              ("43-John.txt",            43),
    "Acts":              ("44-Acts.txt",            44),
    "Romans":            ("45-Romans.txt",          45),
    "I Corinthians":     ("46-1Corinthians.txt",    46),
    "II Corinthians":    ("47-2Corinthians.txt",    47),
    "Galatians":         ("48-Galatians.txt",        48),
    "Ephesians":         ("49-Ephesians.txt",        49),
    "Philippians":       ("50-Philippians.txt",      50),
    "Colossians":        ("51-Colossians.txt",       51),
    "I Thessalonians":   ("52-1Thessalonians.txt",   52),
    "II Thessalonians":  ("53-2Thessalonians.txt",   53),
    "I Timothy":         ("54-1Timothy.txt",         54),
    "II Timothy":        ("55-2Timothy.txt",         55),
    "Titus":             ("56-Titus.txt",            56),
    "Philemon":          ("57-Philemon.txt",         57),
    "Hebrews":           ("58-Hebrews.txt",          58),
    "James":             ("59-James.txt",            59),
    "I Peter":           ("60-1Peter.txt",           60),
    "II Peter":          ("61-2Peter.txt",           61),
    "I John":            ("62-1John.txt",            62),
    "II John":           ("63-2John.txt",            63),
    "III John":          ("64-3John.txt",            64),
    "Jude":              ("65-Jude.txt",             65),
    "Revelation of John":("66-Revelation.txt",       66),
}


def _download_csv(url: str) -> str:
    """Download a URL and return its text content."""
    print(f"Downloading {url} …", flush=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "fetch_kjv_nt/1.0 (ELS research; public domain text)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    print(f"  Downloaded {len(data):,} bytes.")
    return data.decode("utf-8")


def main() -> None:
    texts_dir = Path(__file__).resolve().parent.parent / "texts"
    texts_dir.mkdir(exist_ok=True)

    # ── Download the full KJV CSV ─────────────────────────────────────────────
    try:
        raw_csv = _download_csv(_KJV_CSV_URL)
    except Exception as exc:
        sys.exit(f"ERROR: Could not download KJV CSV: {exc}")

    # ── Parse CSV: columns Book,Chapter,Verse,Text ───────────────────────────
    # Accumulate lines per NT book (keyed by the Book-column string)
    book_lines: dict[str, list[str]] = {bname: [] for bname in _NT_BOOKS}

    reader = csv.reader(io.StringIO(raw_csv))
    header = next(reader, None)  # skip header row (Book,Chapter,Verse,Text)
    print(f"  CSV header: {header}")

    rows_read = 0
    for row in reader:
        if len(row) < 4:
            continue
        book_name_csv = row[0].strip()
        if book_name_csv not in _NT_BOOKS:
            continue  # skip OT rows
        try:
            c = int(row[1])
            v = int(row[2])
        except ValueError:
            continue
        text = row[3].strip()
        if not text:
            continue
        fname, bnum = _NT_BOOKS[book_name_csv]
        book_lines[book_name_csv].append(f"{bnum} {c} {v} {text}")
        rows_read += 1

    print(f"  Parsed {rows_read:,} NT verses from CSV.")

    # ── Write one file per NT book ────────────────────────────────────────────
    total_alpha = 0
    for book_name_csv, (fname, bnum) in _NT_BOOKS.items():
        lines = book_lines[book_name_csv]
        if not lines:
            print(f"  WARNING: no verses found for {book_name_csv}")
            continue
        dest = texts_dir / fname
        content = "\n".join(lines) + "\n"
        dest.write_text(content, encoding="utf-8")

        # Count alpha characters (matches what _parse_kjv_txt will extract)
        alpha_count = sum(
            ch.isalpha()
            for line in lines
            for ch in line.split(maxsplit=3)[-1]
        )
        total_alpha += alpha_count
        print(f"  {fname:30s}  {len(lines):5d} verses  {alpha_count:7,} alpha chars")

    print(f"\nTotal NT alpha characters: {total_alpha:,}")
    print("Done.  Re-run  uv run data_loader.py  to rebuild the corpus.")


if __name__ == "__main__":
    main()
