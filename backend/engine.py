"""
engine.py – High-speed ELS (Equidistant Letter Sequence) search engine.

Strategy
--------
1. On construction, ``data_loader.get_or_build()`` loads (or builds) the
   full concatenated Torah text and its Global Index Map (Polars DataFrame).
2. The full text is compacted to 1-byte-per-letter so StringZilla byte offsets
   equal Hebrew character positions.
3. The index map columns are pre-extracted to plain Python lists so every
   provenance lookup (book name, verse reference) is an O(1) list index.
4. For each skip value *d* (positive = forward / L→R, negative = backward / R→L):
   - A PyTorch strided view is built (tensor is flipped first for backward reads
     because PyTorch does not support negative strides).
   - StringZilla scans the strided byte sequence in hardware-accelerated SIMD.
5. Each hit is enriched with:
   - ``book``     – book name of the first matched letter
   - ``verse``    – verse reference of the first matched letter
   - ``sequence`` – the actual Hebrew letters decoded from the corpus
"""

from __future__ import annotations

import concurrent.futures as _futures
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Sequence

import torch
from stringzilla import Str as SZStr
from tqdm import tqdm

import data_loader as _dl
import validator as _val

# ── Dual-alphabet compact encoding ───────────────────────────────────────────
# Hebrew Unicode block starts at U+05D0 (alef).  Each consonant is mapped to a
# single byte value via ord(c) − 0x05D0 + 1.  Because the block (U+05D0–U+05EA)
# contains both final and non-final forms, after normalisation the non-final
# consonant bytes span 1–27 (alef=1 … tav=27; final-form positions 11,14,16,20,22
# are gaps that never appear in the normalised corpus).
# English letters A–Z are mapped to bytes 28–53 (_ENG_COMPACT_OFFSET+1 … +26),
# which is ENTIRELY ABOVE the Hebrew range — no overlap, no cross-contamination.
_HEB_BASE = 0x05D0
_ENG_COMPACT_OFFSET = 27   # A=28, B=29, …, Z=53

# Final-form → non-final normalization applied to user-supplied search terms
# before compaction.  The corpus text is already normalized by data_loader,
# but user input may contain final consonant forms (ך ם ן ף ץ).
_FINAL_NORM = str.maketrans({
    "\u05DA": "\u05DB",  # ך → כ
    "\u05DD": "\u05DE",  # ם → מ
    "\u05DF": "\u05E0",  # ן → נ
    "\u05E3": "\u05E4",  # ף → פ
    "\u05E5": "\u05E6",  # ץ → צ
})

# KJV New-Testament book names — used by archiver.py to keep ChromaDB Hebrew-only.
# The validation loop no longer gates on this set; score_match() routes to the
# appropriate model (HeBERT for Hebrew, English BERT for KJV) automatically.
_KJV_NT_BOOK_NAMES: frozenset[str] = frozenset({
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "I Corinthians", "II Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "I Thessalonians", "II Thessalonians",
    "I Timothy", "II Timothy", "Titus", "Philemon", "Hebrews",
    "James", "I Peter", "II Peter", "I John", "II John", "III John",
    "Jude", "Revelation",
})


def _compact(text: str) -> bytes:
    """
    Encode a Hebrew or English string to 1-byte-per-letter compact form.

    Hebrew consonants (U+05D0–U+05EA) → bytes 1–27 (offset from alef).
    English letters A–Z (upper or lower case) → bytes 28–53.
    All other characters are silently dropped.

    Final consonant forms (ך ם ן ף ץ) are normalised to their non-final
    equivalents before encoding so that user-supplied search terms match the
    already-normalised corpus bytes produced by data_loader.
    """
    text = text.translate(_FINAL_NORM)
    result: list[int] = []
    for c in text:
        o = ord(c)
        if 0x05D0 <= o <= 0x05EA:   # Hebrew consonant
            result.append(o - _HEB_BASE + 1)
        elif 65 <= o <= 90:          # Uppercase A–Z
            result.append(o - 65 + _ENG_COMPACT_OFFSET + 1)
        elif 97 <= o <= 122:         # Lowercase a–z (auto-uppercase internally)
            result.append(o - 97 + _ENG_COMPACT_OFFSET + 1)
    return bytes(result)


def _decode_byte(b: int) -> str:
    """Decode one compact byte back to its Hebrew Unicode or ASCII character."""
    if b > _ENG_COMPACT_OFFSET:           # English letter (28–53)
        return chr(b - _ENG_COMPACT_OFFSET - 1 + 65)   # → ‘A’–‘Z’
    return chr(b + _HEB_BASE - 1)          # Hebrew letter (1–22)


# ── Dynamic skip scaling ──────────────────────────────────────────────────────

def effective_max_skip(
    words: Sequence[str],
    max_skip: int,
    corpus_len: int,
    *,
    length_threshold: int = 4,
    scale_to: int = 10_000,
) -> int:
    """
    Return an adjusted max_skip that accounts for the reduced hit probability
    of long words in the corpus.

    For any word strictly longer than *length_threshold* letters the skip
    ceiling is raised to at least *scale_to* (default 10 000) so that the
    search covers a statistically meaningful portion of the corpus.  For
    shorter words the caller-supplied *max_skip* is returned unchanged.

    The default threshold of 4 means words of 5 or more letters trigger
    scaling.  5-letter search terms (e.g. טראמפ / דונלד) are rare enough
    in a 1.2 M-letter corpus that the default 1 000-skip ceiling covers only
    a narrow fraction of the probability space; raising to 10 000+ is required
    for statistically meaningful coverage.

    The result is capped at ``corpus_len // 2`` to prevent impractical
    runtimes; in practice this cap is only ever reached for very short words
    in very small sub-corpora.

    Parameters
    ----------
    words :
        The Hebrew search terms (same list passed to ``ELSEngine.search``).
    max_skip :
        The user-requested maximum skip distance.
    corpus_len :
        Total letter count of the corpus (or the filtered sub-corpus being
        searched).  Used for the upper cap.
    length_threshold :
        Words strictly longer than this value trigger scaling (default 4,
        i.e., words of 5+ letters are scaled).
    scale_to :
        Minimum effective max_skip when scaling is triggered (default 10 000).
    """
    if not words:
        return max_skip
    max_word_len = max(len(w) for w in words)
    if max_word_len > length_threshold:
        scaled = max(max_skip, scale_to)
        return min(scaled, max(max_skip, corpus_len // 2))
    return max_skip


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Match:
    word: str            # the search term (Hebrew Unicode)
    skip: int            # ELS skip distance (positive = L→R, negative = R→L)
    start: int           # letter index of the first character in the full corpus text
    length: int          # number of letters in the match
    sequence: str        # actual Hebrew letters decoded from the corpus (Unicode)
    book: str            # book name of the first matched letter, e.g. "Genesis"
    verse: str           # verse reference of the first matched letter, e.g. "Genesis 1:1"
    hebert_score: float  # cosine similarity vs. surrounding context (0.0 if not scored)
    z_score: float = 0.0          # corpus-level Z: (real_hits − μ_shuffle) / σ_shuffle
    is_significant: bool = False  # True when z_score > threshold (default 3.0)
    semantic_z_score: float = 0.0 # per-match: (hebert_score − μ_rand_score) / σ_rand_score


# ── Engine ────────────────────────────────────────────────────────────────────

@dataclass
class ELSEngine:
    """
    High-speed ELS engine backed by the Koren Torah corpus.

    Parameters
    ----------
    min_skip : int
        Minimum |skip| value to search (default 1 = plain consecutive text).
    max_skip : int
        Maximum |skip| value; both +d and −d directions are searched.
    texts_dir : str | Path
        Directory containing the Koren Torah .txt source files.
    data_dir : str | Path
        Directory where ``full_text.bin`` and ``index_map.parquet`` are cached.
    show_load_progress : bool
        Show a tqdm bar when building the corpus for the first time.
    """

    min_skip: int = 1
    max_skip: int = 1000
    texts_dir: str | Path = "texts"
    data_dir: str | Path = "data"
    show_load_progress: bool = True
    validate: bool = False  # when True, score every match with HeBERT
    threads: int = 1        # ThreadPool worker count for skip-loop parallelism
    long_skip: bool = False  # when True, filter out abs(skip) < 10 to isolate hidden codes

    def __post_init__(self) -> None:
        # get_or_build() ensures the cached artifacts exist on disk.
        # We then read the raw text directly because SZStr wraps UTF-8 bytes
        # internally; iterating it does NOT yield Python str characters.
        _dl.get_or_build(
            texts_dir=self.texts_dir,
            output_dir=self.data_dir,
            show_progress=self.show_load_progress,
        )
        bin_path = Path(self.data_dir) / "full_text.bin"
        _, index_map = _dl.load(self.data_dir)
        full_text: str = bin_path.read_bytes().decode("utf-8")

        self._compact_bytes: bytes = _compact(full_text)
        self._char_tensor: torch.Tensor = torch.tensor(
            list(self._compact_bytes), dtype=torch.uint8
        )
        self._text_len: int = len(self._compact_bytes)
        # Pre-materialise provenance columns as Python lists for O(1) per-hit lookup
        self._books: list[str] = index_map["book_name"].to_list()
        self._verses: list[str] = index_map["original_verse_reference"].to_list()

        # Pre-build a mapping of verse_reference → (start_idx, end_idx) for fast
        # context extraction.  We walk the verse list once at init time.
        self._verse_spans: dict[str, tuple[int, int]] = {}
        _prev_ref = self._verses[0]
        _span_start = 0
        for _i, _ref in enumerate(self._verses):
            if _ref != _prev_ref:
                self._verse_spans[_prev_ref] = (_span_start, _i)
                _prev_ref = _ref
                _span_start = _i
        self._verse_spans[_prev_ref] = (_span_start, self._text_len)
        # Ordered list of verse keys for neighbour lookup
        self._verse_order: list[str] = list(dict.fromkeys(self._verses))

        if self.validate:
            _val.warm_up()

        # Timing accumulators (updated after each search() call)
        self.last_simd_secs:      float = 0.0
        self.last_validate_secs:  float = 0.0
        self.last_validate_calls: int   = 0
        self.last_wall_secs:      float = 0.0
        self.last_cpu_secs:       float = 0.0
        self.last_threads:        int   = max(1, int(self.threads))

    @property
    def corpus_bytes(self) -> bytes:
        """The compact 1-byte-per-letter representation of the full Torah corpus."""
        return self._compact_bytes

    def corpus_bytes_for(self, books: Sequence[str] | None = None) -> bytes:
        """
        Return compact bytes scoped to a subset of books.

        Pass the same book filter used for the real search so that the Monte
        Carlo baseline is computed on the same letter population.
        """
        if books is None:
            return self._compact_bytes
        book_set = set(books)
        return bytes(
            b for b, bk in zip(self._compact_bytes, self._books)
            if bk in book_set
        )

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Context-window helper
    # ------------------------------------------------------------------

    def _context_for(self, verse_ref: str) -> str:
        """
        Return the plain Hebrew text of the verse before, the matched verse,
        and the verse after *verse_ref* concatenated as a single string.
        The text is decoded from the compact byte store so no re-reading is needed.
        """
        order = self._verse_order
        try:
            idx = order.index(verse_ref)
        except ValueError:
            idx = 0
        neighbours = order[max(0, idx - 1) : idx + 2]  # up to 3 verses
        parts: list[str] = []
        for ref in neighbours:
            s, e = self._verse_spans[ref]
            parts.append("".join(_decode_byte(b) for b in self._compact_bytes[s:e]))
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        words: Sequence[str],
        *,
        books: Sequence[str] | None = None,
        show_progress: bool = True,
        threads: int | None = None,
    ) -> list[Match]:
        """
        Search for one or more Hebrew *or English* words across the full corpus
        (Tanakh + KJV NT when available) using ELS with all skip values in
        ``±[min_skip, max_skip]``.

        Hebrew search terms (bytes 1–22) only match Tanakh sections.
        English search terms (bytes 23–48) only match KJV NT sections.
        Both types can be mixed in one call.

        Parameters
        ----------
        words :
            One or more Hebrew Unicode or uppercase English strings to search for.
        books :
            Optional list of book names to restrict the search
            (e.g. ``["Genesis", "Deuteronomy"]`` or ``["Matthew", "John"]``).
            ``None`` searches all books.
        show_progress :
            Show a tqdm progress bar over skip iterations.
        threads :
            Number of ThreadPool workers to use for skip-chunk parallelism.
            ``None`` uses ``self.threads``.

        Returns
        -------
        list[Match]
        """
        compact_words = [_compact(w) for w in words]
        allowed_books: set[str] | None = set(books) if books else None

        skips = list(range(self.min_skip, self.max_skip + 1))
        skips += [-d for d in range(self.min_skip, self.max_skip + 1)]

        max_workers = max(1, int(self.threads if threads is None else threads))
        validate_lock = Lock()

        # Pre-warm word embeddings once so every per-match call hits the cache.
        # Guard with a lock because validator's embed cache is a shared mutable dict.
        # Route to the correct model to avoid a wasted HeBERT call for English words.
        if self.validate:
            with validate_lock:
                for w in words:
                    if _val._is_english_text(w):
                        _val._embed_english(w)
                    else:
                        _val._embed(w)

        def _scan_skip_batch(skip_batch: Sequence[int]) -> tuple[list[Match], float, int]:
            local_results: list[Match] = []
            local_validate_total = 0.0
            local_validate_calls = 0

            for skip in skip_batch:
                abs_skip = abs(skip)
                if abs_skip >= self._text_len:
                    continue

                if skip > 0:
                    strided_vals = self._char_tensor[::skip].tolist()

                    def orig_pos(i: int, s: int = skip) -> int:
                        return i * s
                else:
                    strided_vals = self._char_tensor.flip(0)[::abs_skip].tolist()

                    def orig_pos(i: int, s: int = abs_skip) -> int:  # noqa: E731
                        return (self._text_len - 1) - i * s

                strided_bytes = bytes(strided_vals)
                sz_str = SZStr(strided_bytes)

                for word, cword in zip(words, compact_words):
                    wlen = len(cword)
                    pos = 0
                    while True:
                        idx = sz_str.find(cword, pos)
                        if idx < 0:
                            break

                        start = orig_pos(idx)
                        book = self._books[start]
                        if allowed_books is not None and book not in allowed_books:
                            pos = idx + 1
                            continue

                        verse = self._verses[start]
                        sequence = "".join(
                            _decode_byte(self._compact_bytes[start + j * skip])
                            for j in range(wlen)
                        )

                        if self.validate:
                            t0 = time.perf_counter()
                            context = self._context_for(verse)
                            with validate_lock:
                                hebert_score = _val.score_match(word, context)
                            local_validate_total += time.perf_counter() - t0
                            local_validate_calls += 1
                        else:
                            hebert_score = 0.0

                        local_results.append(
                            Match(
                                word=word,
                                skip=skip,
                                start=start,
                                length=wlen,
                                sequence=sequence,
                                book=book,
                                verse=verse,
                                hebert_score=hebert_score,
                            )
                        )
                        pos = idx + 1

            return local_results, local_validate_total, local_validate_calls

        def _chunked(data: list[int], chunk_size: int) -> list[list[int]]:
            return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        chunk_size = max(1, len(skips) // max(1, max_workers * 8))
        skip_batches = _chunked(skips, chunk_size)

        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        results: list[Match] = []
        t_validate_total = 0.0
        n_validate = 0

        with tqdm(total=len(skips), desc="ELS search", disable=not show_progress) as pbar:
            if max_workers == 1 or len(skip_batches) == 1:
                for batch in skip_batches:
                    batch_results, batch_t_validate, batch_n_validate = _scan_skip_batch(batch)
                    results.extend(batch_results)
                    t_validate_total += batch_t_validate
                    n_validate += batch_n_validate
                    pbar.update(len(batch))
            else:
                with _futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    future_sizes = {
                        ex.submit(_scan_skip_batch, batch): len(batch) for batch in skip_batches
                    }
                    for fut in _futures.as_completed(future_sizes):
                        batch_results, batch_t_validate, batch_n_validate = fut.result()
                        results.extend(batch_results)
                        t_validate_total += batch_t_validate
                        n_validate += batch_n_validate
                        pbar.update(future_sizes[fut])

        wall_total = time.perf_counter() - wall_start
        cpu_total = time.process_time() - cpu_start
        t_simd = max(wall_total - t_validate_total, 0.0)

        self.last_simd_secs = t_simd
        self.last_validate_secs = t_validate_total
        self.last_validate_calls = n_validate
        self.last_wall_secs = wall_total
        self.last_cpu_secs = cpu_total
        self.last_threads = max_workers

        if self.long_skip:
            results = [m for m in results if abs(m.skip) >= 10]

        n_matches = len(results)
        simd_rate = n_matches / t_simd if t_simd > 0 else float("inf")
        cpu_eff = cpu_total / wall_total if wall_total > 0 else 0.0
        perf_msg = (
            f"  SIMD: {t_simd:.3f}s  ({simd_rate:.0f} matches/s)"
            f"  |  Wall-clock: {wall_total:.3f}s  |  CPU time: {cpu_total:.3f}s"
            f"  |  Threads: {max_workers}  |  CPU/Wall: {cpu_eff:.2f}x"
        )
        if self.validate and n_validate > 0:
            val_rate = n_validate / t_validate_total if t_validate_total > 0 else 0.0
            cache_info = _val.embed_cache_stats()
            perf_msg += (
                f"  |  HeBERT: {t_validate_total:.3f}s  "
                f"({val_rate:.1f} calls/s, "
                f"cache {cache_info['hits']} hits / {cache_info['misses']} misses)"
            )
        print(perf_msg)

        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def search_single(self, word: str, **kwargs) -> list[Match]:
        """Shorthand for searching a single word."""
        return self.search([word], **kwargs)
