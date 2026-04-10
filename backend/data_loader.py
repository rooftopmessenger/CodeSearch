"""
data_loader.py – Load and prepare the full Tanakh text files for ELS searching.

Sources
-------
Torah (5 books):
    Koren .txt files using BHS-style ASCII transliteration (e.g. B=ב, $=ש, )=א).
    Each line: book_num  chapter  verse  WORD1 WORD2 ...

Nevi’im + Ketuvim (34 books):
    Leningrad Codex .json files (Hebrew Unicode, consonant-only text, no nikud).
    Structure: { "text": [[verse_str, ...], ...] }  (chapters × verses).

Encoding convention
-------------------
All 22 standard Hebrew letters are encoded as compact 1-byte values
(ord(c) − 0x05D0 + 1).  Final letter forms (ךםןףץ) are normalised to their
non-final counterparts before encoding so that the 22-letter ELS alphabet
is consistently maintained across both source formats.

This module:
  1. Reads all 39 books in canonical Tanakh order (Torah → Nevi’im → Ketuvim).
  2. Normalises and filters each letter to the 22-letter Hebrew alphabet.
  3. Builds a Polars DataFrame (“Global Index Map”): global_index | char | book_name | original_verse_reference
  4. Saves the concatenated Hebrew string to  data/full_text.bin  (UTF-8).
  5. Saves the index map to  data/index_map.parquet  for instant re-loading.

Run directly to build the artifacts:
    uv run data_loader.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from stringzilla import Str as SZStr
from tqdm import tqdm

# ── Canonical Tanakh book order (39 books) ───────────────────────────────
# Each entry: (filename, canonical_book_name, source_format)
# source_format: "koren" = BHS ASCII .txt   |   "leningrad" = Hebrew Unicode .json
_TANAKH_BOOKS: list[tuple[str, str, str]] = [
    # ── Torah ────────────────────────────────────────────────────────────────
    ("text_koren_1genesis.txt",           "Genesis",       "koren"),
    ("text_koren_2exodus.txt",            "Exodus",        "koren"),
    ("text_koren_3leviticus.txt",         "Leviticus",     "koren"),
    ("text_koren_4numbers.txt",           "Numbers",       "koren"),
    ("text_koren_5deuteronomy.txt",       "Deuteronomy",   "koren"),
    # ── Nevi’im Rishonim (Former Prophets) ──────────────────────────────────
    ("text_leningrad_6joshua.json",       "Joshua",        "leningrad"),
    ("text_leningrad_7judges.json",       "Judges",        "leningrad"),
    ("text_leningrad_8Isamuel.json",      "I Samuel",      "leningrad"),
    ("text_leningrad_9IIsamuel.json",     "II Samuel",     "leningrad"),
    ("text_leningrad_10Ikings.json",      "I Kings",       "leningrad"),
    ("text_leningrad_11IIkings.json",     "II Kings",      "leningrad"),
    # ── Nevi’im Acharonim (Latter Prophets) ──────────────────────────────
    ("text_leningrad_12isaiah.json",      "Isaiah",        "leningrad"),
    ("text_leningrad_13jeremiah.json",    "Jeremiah",      "leningrad"),
    ("text_leningrad_14ezekiel.json",     "Ezekiel",       "leningrad"),
    # The Twelve (Trei Asar)
    ("text_leningrad_15hosea.json",       "Hosea",         "leningrad"),
    ("text_leningrad_16joel.json",        "Joel",          "leningrad"),
    ("text_leningrad_17amos.json",        "Amos",          "leningrad"),
    ("text_leningrad_18obadiah.json",     "Obadiah",       "leningrad"),
    ("text_leningrad_19jonah.json",       "Jonah",         "leningrad"),
    ("text_leningrad_20micah.json",       "Micah",         "leningrad"),
    ("text_leningrad_21nahum.json",       "Nahum",         "leningrad"),
    ("text_leningrad_22habakkuk.json",    "Habakkuk",      "leningrad"),
    ("text_leningrad_23zephaniah.json",   "Zephaniah",     "leningrad"),
    ("text_leningrad_24haggai.json",      "Haggai",        "leningrad"),
    ("text_leningrad_25zechariah.json",   "Zechariah",     "leningrad"),
    ("text_leningrad_26malachi.json",     "Malachi",       "leningrad"),
    # ── Ketuvim (Writings) ───────────────────────────────────────────────────────
    ("text_leningrad_27psalms.json",      "Psalms",        "leningrad"),
    ("text_leningrad_28proverbs.json",    "Proverbs",      "leningrad"),
    ("text_leningrad_29job.json",         "Job",           "leningrad"),
    ("text_leningrad_30songofsongs.json", "Song of Songs", "leningrad"),
    ("text_leningrad_31ruth.json",        "Ruth",          "leningrad"),
    ("text_leningrad_32lamentations.json","Lamentations",  "leningrad"),
    ("text_leningrad_33ecclesiastes.json","Ecclesiastes",  "leningrad"),
    ("text_leningrad_34esther.json",      "Esther",        "leningrad"),
    ("text_leningrad_35daniel.json",      "Daniel",        "leningrad"),
    ("text_leningrad_36ezra.json",        "Ezra",          "leningrad"),
    ("text_leningrad_37nehemiah.json",    "Nehemiah",      "leningrad"),
    ("text_leningrad_38Ichronicles.json", "I Chronicles",  "leningrad"),
    ("text_leningrad_39IIchronicles.json","II Chronicles", "leningrad"),
]

# ── New Testament (KJV) book order (27 books) ─────────────────────────────────
# source_format: "kjv" = plain-text .txt, same column layout as Koren:
#   book_num  chapter  verse  word1 word2 ...
# Only alphabetic characters are extracted and stored as uppercase ASCII.
# Files are OPTIONAL — missing files produce a warning, not an error.
_KJV_NT_BOOKS: list[tuple[str, str, str]] = [
    ("40-Matthew.txt",         "Matthew",           "kjv"),
    ("41-Mark.txt",            "Mark",               "kjv"),
    ("42-Luke.txt",            "Luke",               "kjv"),
    ("43-John.txt",            "John",               "kjv"),
    ("44-Acts.txt",            "Acts",               "kjv"),
    ("45-Romans.txt",          "Romans",             "kjv"),
    ("46-1Corinthians.txt",    "I Corinthians",      "kjv"),
    ("47-2Corinthians.txt",    "II Corinthians",     "kjv"),
    ("48-Galatians.txt",       "Galatians",          "kjv"),
    ("49-Ephesians.txt",       "Ephesians",          "kjv"),
    ("50-Philippians.txt",     "Philippians",        "kjv"),
    ("51-Colossians.txt",      "Colossians",         "kjv"),
    ("52-1Thessalonians.txt",  "I Thessalonians",    "kjv"),
    ("53-2Thessalonians.txt",  "II Thessalonians",   "kjv"),
    ("54-1Timothy.txt",        "I Timothy",          "kjv"),
    ("55-2Timothy.txt",        "II Timothy",         "kjv"),
    ("56-Titus.txt",           "Titus",              "kjv"),
    ("57-Philemon.txt",        "Philemon",           "kjv"),
    ("58-Hebrews.txt",         "Hebrews",            "kjv"),
    ("59-James.txt",           "James",              "kjv"),
    ("60-1Peter.txt",          "I Peter",            "kjv"),
    ("61-2Peter.txt",          "II Peter",           "kjv"),
    ("62-1John.txt",           "I John",             "kjv"),
    ("63-2John.txt",           "II John",            "kjv"),
    ("64-3John.txt",           "III John",           "kjv"),
    ("65-Jude.txt",            "Jude",               "kjv"),
    ("66-Revelation.txt",      "Revelation",         "kjv"),
]

# ── BHS ASCII → Hebrew Unicode consonant mapping (non-final forms throughout) ─
_TRANS_MAP: dict[str, str] = {
    ")": "\u05D0",  # א alef
    "B": "\u05D1",  # ב bet
    "G": "\u05D2",  # ג gimel
    "D": "\u05D3",  # ד dalet
    "H": "\u05D4",  # ה hey
    "W": "\u05D5",  # ו vav
    "Z": "\u05D6",  # ז zayin
    "X": "\u05D7",  # ח chet
    "+": "\u05D8",  # ט tet
    "Y": "\u05D9",  # י yod
    "K": "\u05DB",  # כ kaf  (non-final; ך is not separated in ELS tradition)
    "L": "\u05DC",  # ל lamed
    "M": "\u05DE",  # מ mem  (non-final)
    "N": "\u05E0",  # נ nun  (non-final)
    "S": "\u05E1",  # ס samech
    "(": "\u05E2",  # ע ayin
    "P": "\u05E4",  # פ pe   (non-final)
    "C": "\u05E6",  # צ tsade (non-final)
    "Q": "\u05E7",  # ק kuf
    "R": "\u05E8",  # ר resh
    "$": "\u05E9",  # ש shin
    "T": "\u05EA",  # ת tav
}

# Pre-compile a str.translate table for fast per-character conversion
_TRANS_TABLE = str.maketrans(_TRANS_MAP)

# Valid Hebrew Unicode range for the normalization guard
_HEB_MIN = "\u05D0"
_HEB_MAX = "\u05EA"

# Final form → non-final normalization for ELS 22-letter alphabet convention.
# In ELS tradition ך/כ, ם/מ, ן/נ, ף/פ, ץ/צ are treated as the same letter.
# Leningrad Codex JSON uses final forms; Koren ASCII uses non-final only.
_FINAL_TO_NONFINAL = str.maketrans({
    "\u05DA": "\u05DB",  # ך final kaf   → כ kaf
    "\u05DD": "\u05DE",  # ם final mem   → מ mem
    "\u05DF": "\u05E0",  # ן final nun   → נ nun
    "\u05E3": "\u05E4",  # ף final pe    → פ pe
    "\u05E5": "\u05E6",  # ץ final tsade → צ tsade
})


def _is_hebrew(ch: str) -> bool:
    return _HEB_MIN <= ch <= _HEB_MAX


def _parse_book(path: Path, book_name: str) -> tuple[list[str], list[str], list[str]]:
    """
    Read one Koren .txt file and return three parallel lists:
        chars       – Hebrew Unicode letters (one element per letter)
        books       – book_name repeated for every letter
        verse_refs  – "Book chapter:verse" string for every letter
    """
    chars: list[str] = []
    books: list[str] = []
    verse_refs: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        # Each non-empty line must begin with: book_num chapter verse …text…
        if len(parts) < 4:
            continue
        try:
            chapter = int(parts[1])
            verse   = int(parts[2])
        except ValueError:
            continue  # skip malformed header lines

        verse_ref = f"{book_name} {chapter}:{verse}"
        text_tokens = parts[3:]  # the transliterated words for this line

        for token in text_tokens:
            for ascii_ch in token:
                hebrew_ch = _TRANS_MAP.get(ascii_ch)
                if hebrew_ch is None:
                    continue  # skip spaces, punctuation, or unknown chars
                # Normalisation guard: only emit characters in the Hebrew block
                if not _is_hebrew(hebrew_ch):
                    continue
                chars.append(hebrew_ch)
                books.append(book_name)
                verse_refs.append(verse_ref)

    return chars, books, verse_refs

def _parse_leningrad_json(
    path: Path, book_name: str
) -> tuple[list[str], list[str], list[str]]:
    """
    Parse a Leningrad Codex JSON file and return three parallel lists
    identical in structure to those from :func:`_parse_book`.

    The JSON format is::

        { "text": [[verse_str, ...], ...] }   # outer = chapters, inner = verses

    Each verse string is in Hebrew Unicode.  Final letter forms are normalised
    to their non-final counterparts before being emitted, so both sources
    produce a consistent 22-letter compact alphabet.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    chapters = data["text"]

    chars: list[str] = []
    books: list[str] = []
    verse_refs: list[str] = []

    for chap_idx, chapter in enumerate(chapters, start=1):
        if not isinstance(chapter, list):
            continue
        for verse_idx, verse in enumerate(chapter, start=1):
            # Some books nest verses as lists of half-verse strings
            if isinstance(verse, list):
                verse_text = " ".join(v for v in verse if isinstance(v, str))
            elif isinstance(verse, str):
                verse_text = verse
            else:
                continue
            if not verse_text:
                continue

            verse_ref = f"{book_name} {chap_idx}:{verse_idx}"
            # Normalise final forms then filter to 22 Hebrew consonants
            normalized = verse_text.translate(_FINAL_TO_NONFINAL)
            for ch in normalized:
                if _is_hebrew(ch):
                    chars.append(ch)
                    books.append(book_name)
                    verse_refs.append(verse_ref)

    return chars, books, verse_refs

def _parse_kjv_txt(
    path: Path, book_name: str
) -> tuple[list[str], list[str], list[str]]:
    """
    Parse a KJV plain-text file and return three parallel lists identical in
    structure to those from :func:`_parse_book`.

    Expected line format::

        book_num  chapter  verse  word1 word2 ...

    Only alphabetic characters are extracted; each letter is stored as an
    uppercase ASCII character (A–Z).  Spaces, punctuation, and numbers are
    silently discarded.  The compact byte encoding (A=28 … Z=53) is applied
    later by :func:`engine._compact` to keep the two alphabets non-overlapping.
    """
    chars: list[str] = []
    books: list[str] = []
    verse_refs: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 4:
            continue
        try:
            chapter = int(parts[1])
            verse   = int(parts[2])
        except ValueError:
            continue

        verse_ref = f"{book_name} {chapter}:{verse}"
        for token in parts[3:]:
            for ch in token:
                if ch.isalpha():
                    chars.append(ch.upper())
                    books.append(book_name)
                    verse_refs.append(verse_ref)

    return chars, books, verse_refs

# ── Public API ─────────────────────────────────────────────────────────────────

def build(
    texts_dir: Path | str = Path(__file__).resolve().parent.parent / "texts",
    output_dir: Path | str = Path(__file__).resolve().parent.parent / "data",
    *,
    show_progress: bool = True,
) -> tuple[SZStr, pl.DataFrame]:
    """
    Build the full-text string and Global Index Map from all available books.

    Tanakh (39 books, always required):
        Torah from Koren .txt (BHS ASCII transliteration);
        Nevi'im + Ketuvim from Leningrad Codex .json (Hebrew Unicode).

    New Testament / KJV (27 books, optional):
        Plain .txt files named 40-Matthew.txt … 66-Revelation.txt.
        Missing NT files are skipped with a warning rather than raising.

    Encoding: Hebrew letters → bytes 1–27; English letters → bytes 28–53.
    These disjoint ranges ensure Hebrew search patterns never match English
    text and vice versa (the compact encoding is performed by engine._compact).

    Returns
    -------
    full_sz : SZStr
        StringZilla-wrapped concatenated text (Hebrew Unicode + ASCII English).
    index_map : pl.DataFrame
        Columns: global_index (UInt32), char (Utf8),
                 book_name (Utf8), original_verse_reference (Utf8).
    """
    texts_dir = Path(texts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chars:  list[str] = []
    all_books:  list[str] = []
    all_refs:   list[str] = []

    files = [
        (texts_dir / fname, book, src)
        for fname, book, src in _TANAKH_BOOKS
    ]
    missing = [str(p) for p, _, _ in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Tanakh text file(s): {', '.join(missing)}"
        )

    for path, book_name, source in tqdm(
        files, desc="Loading books", disable=not show_progress
    ):
        if source == "koren":
            chars, books, refs = _parse_book(path, book_name)
        else:
            chars, books, refs = _parse_leningrad_json(path, book_name)
        all_chars.extend(chars)
        all_books.extend(books)
        all_refs.extend(refs)

    # ── KJV New Testament (optional) ──────────────────────────────────────────
    # Each NT book is gracefully skipped if its file is absent, so existing
    # Tanakh-only workflows are never broken.
    n_tanakh = len(all_chars)  # snapshot before NT loading
    n_kjv_loaded = 0
    for fname, book_name, source in _KJV_NT_BOOKS:
        kjv_path = texts_dir / fname
        if not kjv_path.exists():
            print(f"  [KJV] {fname} not found — skipping.", file=sys.stderr)
            continue
        chars, books, refs = _parse_kjv_txt(kjv_path, book_name)
        all_chars.extend(chars)
        all_books.extend(books)
        all_refs.extend(refs)
        n_kjv_loaded += 1
    if n_kjv_loaded:
        n_nt = len(all_chars) - n_tanakh
        print(f"  [KJV] Loaded {n_kjv_loaded}/27 NT books — {n_nt:,} English letters.",
              file=sys.stderr)

    # ── Concatenate with StringZilla ──────────────────────────────────────────
    # SZStr is a search type; we build the Python str first, then wrap it.
    full_text: str = "".join(all_chars)
    full_sz = SZStr(full_text)

    # ── Save full_text.bin (UTF-8 bytes) ─────────────────────────────────────
    bin_path = output_dir / "full_text.bin"
    bin_path.write_bytes(full_text.encode("utf-8"))

    # ── Build Polars Global Index Map ─────────────────────────────────────────
    n = len(all_chars)
    index_map = pl.DataFrame(
        {
            "global_index":            pl.Series(range(n), dtype=pl.UInt32),
            "char":                    pl.Series(all_chars, dtype=pl.Utf8),
            "book_name":               pl.Series(all_books, dtype=pl.Utf8),
            "original_verse_reference": pl.Series(all_refs, dtype=pl.Utf8),
        }
    )

    # ── Save index_map.parquet ────────────────────────────────────────────────
    parquet_path = output_dir / "index_map.parquet"
    index_map.write_parquet(parquet_path)

    return full_sz, index_map


def load(
    output_dir: Path | str = Path(__file__).resolve().parent.parent / "data",
) -> tuple[SZStr, pl.DataFrame]:
    """
    Fast reload of pre-built artifacts from disk.

    Returns the same (full_sz, index_map) tuple as build().
    Raises FileNotFoundError if the artifacts have not been built yet.
    """
    output_dir = Path(output_dir)
    bin_path    = output_dir / "full_text.bin"
    parquet_path = output_dir / "index_map.parquet"

    for p in (bin_path, parquet_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found – run data_loader.py to build the artifacts first."
            )

    full_text = bin_path.read_bytes().decode("utf-8")
    full_sz   = SZStr(full_text)
    index_map = pl.read_parquet(parquet_path)
    return full_sz, index_map


def get_or_build(
    texts_dir: Path | str = Path(__file__).resolve().parent.parent / "texts",
    output_dir: Path | str = Path(__file__).resolve().parent.parent / "data",
    *,
    show_progress: bool = True,
) -> tuple[SZStr, pl.DataFrame]:
    """Load from cache if available, otherwise build and cache."""
    try:
        return load(output_dir)
    except FileNotFoundError:
        return build(texts_dir, output_dir, show_progress=show_progress)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    full_sz, index_map = build(show_progress=True)
    n = len(index_map)
    n_tanakh = index_map.filter(
        ~pl.col("book_name").is_in([
            "Matthew","Mark","Luke","John","Acts","Romans","I Corinthians",
            "II Corinthians","Galatians","Ephesians","Philippians","Colossians",
            "I Thessalonians","II Thessalonians","I Timothy","II Timothy","Titus",
            "Philemon","Hebrews","James","I Peter","II Peter","I John","II John",
            "III John","Jude","Revelation",
        ])
    ).height
    n_nt = n - n_tanakh
    print(f"\nBuilt corpus: {n:,} total letters  "
          f"({n_tanakh:,} Tanakh Hebrew + {n_nt:,} KJV NT English)")
    print("\nLetter counts per book:")
    counts = (
        index_map
        .group_by("book_name", maintain_order=True)
        .agg(pl.len().alias("letters"))
    )
    for row in counts.iter_rows(named=True):
        print(f"  {row['book_name']:<16} {row['letters']:>7,}")
    _data_dir = Path(__file__).resolve().parent.parent / "data"
    print(f"\nSaved → data/full_text.bin       ({(_data_dir / 'full_text.bin').stat().st_size:,} bytes)")
    print(f"Saved → data/index_map.parquet   ({(_data_dir / 'index_map.parquet').stat().st_size:,} bytes)")
