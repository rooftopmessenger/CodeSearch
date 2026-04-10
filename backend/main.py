"""
main.py – Entry point for the Bible Code ELS searcher.

Usage examples
--------------
  # Search all Torah books, skips 1–100, show top 20:
  uv run main.py --words תורה --max-skip 100 --top 20

  # Search Genesis and Deuteronomy only:
  uv run main.py --words תורה --books Genesis Deuteronomy --max-skip 100

  # Search with HeBERT semantic scoring, keep only matches scoring ≥ 0.60:
  uv run main.py --words תורה --max-skip 200 --validate --min-score 0.60

  # Run Monte Carlo baseline (100 trials) and flag statistically significant hits:
  uv run main.py --words תורה --max-skip 50 --baseline

  # Combine scoring + baseline, custom trial count and Z threshold:
  uv run main.py --words תורה --max-skip 50 --validate --baseline --baseline-trials 200 --z-threshold 2.5

  # Full pipeline with per-match semantic Z-scores (requires --validate + --baseline):
  uv run main.py --words תורה --books Genesis --max-skip 50 --validate --baseline --sem-sample 500

  # Search multiple words, suppress progress bar:
  uv run main.py --words משה אהרן --max-skip 500 --no-progress

  # Export results to CSV:
  uv run main.py --words תורה --max-skip 100 --output results

  # Export to Parquet with full scoring pipeline:
  uv run main.py --words תורה --validate --baseline --output torah_els --output-format parquet

  # Render colour-coded HTML grid for all matches (auto-detects optimal width 2-2000):
  uv run main.py --words תורה --books Genesis --max-skip 50 --view-grid

  # Use a specific grid width and write the HTML to a named file:
  uv run main.py --words תורה אלהים --max-skip 200 --view-grid --grid-width 613 --grid-output genesis_table

  # Auto-translate English words to Hebrew before searching (uses Google Translate):
  uv run main.py --words Torah Moses --translate --max-skip 200

  # Translate + expand with Hebrew synonyms from static lexicon (default, recommended):
  uv run main.py --words President King --translate --expand --max-skip 1000

  # Expand using legacy Google Translate paraphrase probes (requires network):
  uv run main.py --words President King --translate --expand --expand-method llm --max-skip 1000

  # Expand Hebrew input directly using related roots from lexicon:
  uv run main.py --words נשיא --translate --expand --max-skip 1000

  # Search KJV New Testament directly with English ELS (no translation needed):
  uv run main.py --words JESUS --books Matthew --max-skip 50

  # Search English across all KJV NT books:
  uv run main.py --words LOVE --books Matthew Mark Luke John --max-skip 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from engine import ELSEngine, effective_max_skip as _effective_max_skip
import stats as _stats
import reporter as _reporter
import gridder as _gridder
import translator as _translator
import archiver as _archiver

# ── Book-group aliases ────────────────────────────────────────────────────────
# Any key here can be passed to --books and will expand to the listed books.
_BOOK_GROUPS: dict[str, list[str]] = {
    "Torah": [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    ],
    "Neviim": [
        "Joshua", "Judges", "I Samuel", "II Samuel", "I Kings", "II Kings",
        "Isaiah", "Jeremiah", "Ezekiel",
        "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
        "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    ],
    "Ketuvim": [
        "Psalms", "Proverbs", "Job", "Song of Songs", "Ruth", "Lamentations",
        "Ecclesiastes", "Esther", "Daniel", "Ezra", "Nehemiah",
        "I Chronicles", "II Chronicles",
    ],
    "Tanakh": [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        "Joshua", "Judges", "I Samuel", "II Samuel", "I Kings", "II Kings",
        "Isaiah", "Jeremiah", "Ezekiel",
        "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
        "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
        "Psalms", "Proverbs", "Job", "Song of Songs", "Ruth", "Lamentations",
        "Ecclesiastes", "Esther", "Daniel", "Ezra", "Nehemiah",
        "I Chronicles", "II Chronicles",
    ],
    "NT": [
        "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
        "I Corinthians", "II Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "I Thessalonians", "II Thessalonians",
        "I Timothy", "II Timothy", "Titus", "Philemon", "Hebrews",
        "James", "I Peter", "II Peter", "I John", "II John", "III John",
        "Jude", "Revelation",
    ],
}


def _resolve_book_filter(books: list[str] | None) -> list[str] | None:
    """Expand any group-alias names in *books*; deduplicate while preserving order."""
    if not books:
        return books
    seen: set[str] = set()
    resolved: list[str] = []
    for b in books:
        for name in _BOOK_GROUPS.get(b, [b]):
            if name not in seen:
                seen.add(name)
                resolved.append(name)
    return resolved or None


def _auto_threads(num_words: int, skip_range: int) -> int:
    """Return the workload-aware default thread count.

    Uses a single thread for light workloads (≤ 2 search words **and**
    skip range ≤ 20,000), where thread-pool overhead would exceed the
    benefit.  Falls back to the full logical CPU count for heavier jobs.
    """
    if num_words <= 2 and skip_range <= 20_000:
        return 1
    return max(1, os.cpu_count() or 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-speed Bible Code (ELS) searcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--words",
        nargs="+",
        required=True,
        metavar="WORD",
        help="One or more Hebrew or English words to search for. "
             "Hebrew Unicode input searches the Tanakh; uppercase English searches the KJV NT. "
             "Use --translate to auto-convert English to Hebrew before searching.",
    )
    parser.add_argument(
        "--texts-dir",
        default=str(Path(__file__).resolve().parent.parent / "texts"),
        metavar="DIR",
        help="Directory containing the source text files (default: <project-root>/texts/).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent / "data"),
        metavar="DIR",
        help="Directory for cached full_text.bin / index_map.parquet (default: <project-root>/data/).",
    )
    parser.add_argument(
        "--books",
        nargs="*",
        metavar="BOOK",
        help="Limit search to specific book names, e.g. Genesis Exodus Matthew John. "
             "Omit to search all available books.",
    )
    parser.add_argument(
        "--min-skip",
        type=int,
        default=1,
        metavar="N",
        help="Minimum ELS skip distance (default: 1).",
    )
    parser.add_argument(
        "--max-skip",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum ELS skip distance (default: 1000).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "ThreadPool workers for skip-loop parallelism. "
            "Default: automatic — 1 thread when ≤ 2 words and skip range ≤ 20,000; "
            "otherwise all logical CPUs."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="Show only the top N results sorted by |skip| (0 = show all).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run Monte Carlo baseline to calculate Z-scores and flag significant hits.",
    )
    parser.add_argument(
        "--baseline-trials",
        type=int,
        default=100,
        metavar="N",
        help="Number of Monte Carlo shuffle trials (default: 100).",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        metavar="F",
        help="Z-score threshold for is_significant flag (default: 3.0).",
    )
    parser.add_argument(
        "--sem-sample",
        type=int,
        default=500,
        metavar="N",
        help="Max random hit samples scored by HeBERT during baseline run to build the "
             "semantic null distribution (default: 500). Only used when both "
             "--validate and --baseline are active. Set to 0 to disable.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Score every match with HeBERT cosine similarity (slower; Hebrew only).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        metavar="F",
        help="Only keep matches with hebert_score >= F (requires --validate; default: 0.0).",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Auto-translate any English word in --words to Hebrew via Google Translate "
             "before searching.  Hebrew inputs are passed through unchanged (only "
             "final-form normalisation is applied).  The original English is preserved "
             "as a label in the output table.",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="When --translate is active, also fetch up to --expand-count Hebrew synonyms "
             "per translated term and include them as additional search words.",
    )
    parser.add_argument(
        "--expand-count",
        type=int,
        default=2,
        metavar="N",
        help="Number of synonyms to add per translated term when --expand is active "
             "(default: 2).",
    )
    parser.add_argument(
        "--expand-method",
        choices=["lexicon", "llm"],
        default="lexicon",
        help="Synonym source when --expand is active: "
             "'lexicon' (default) uses the static expander.py Biblical Hebrew dictionary — "
             "deterministic, no network, themed around Leader/Messianic/Covenant concepts; "
             "'llm' uses Google Translate paraphrase probes (legacy, unreliable).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress progress bars.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Export all matched results to FILE.csv (or FILE.parquet with "
             "--output-format parquet). Omit to skip export.",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "parquet"],
        default="csv",
        help="Export file format: csv (default) or parquet.",
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Automatically raise max-skip when any search word is longer than 4 letters "
             "(i.e., 5+ letter words trigger scaling). "
             "The skip ceiling is raised to at least the value given by --scale-to "
             "(default 10 000), compensating for the lower ELS hit probability of "
             "long strings in the corpus.",
    )
    parser.add_argument(
        "--scale-to",
        type=int,
        default=10_000,
        metavar="N",
        help="Minimum effective max-skip used when --auto-scale fires (default: 10 000). "
             "For 6-letter words in the full 1.2M Tanakh a value of 50 000 is recommended.",
    )
    parser.add_argument(
        "--view-grid",
        action="store_true",
        help="Render a colour-coded HTML grid for all matches and open it in the "
             "default browser.",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=0,
        metavar="N",
        help="Fix the grid width instead of auto-detecting the optimal width "
             "(2–2000). 0 = auto-detect (default).",
    )
    parser.add_argument(
        "--grid-output",
        default="grid",
        metavar="FILE",
        help="Base filename for the HTML grid output (default: grid → grid.html).",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive significant matches to the ChromaDB vector store (data/chroma/). "
             "Requires --validate and --baseline to be active so that HeBERT embeddings "
             "and corpus Z-scores are available for each match.",
    )
    parser.add_argument(
        "--long-skip",
        action="store_true",
        help="Long-Skip Filter Mode: after searching, discard all matches where "
             "abs(skip) < 10. Eliminates near-plaintext matches and isolates the "
             "hidden ELS signal. The Monte Carlo baseline is also filtered identically "
             "so the null distribution remains comparable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Term preparation (date/year conversion always-on; translation optional) ─
    # prepare_search_terms always runs so that Gregorian dates and 4-digit years
    # are converted to Hebrew automatically regardless of --translate.
    # Google Translate is only used when --translate is explicitly supplied.
    if args.translate:
        print("Auto-translating search terms …")
    prepared = _translator.prepare_search_terms(
        args.words,
        expand=args.expand if args.translate else False,
        expand_count=args.expand_count,
        expand_method=args.expand_method,
        show_translations=not args.no_progress,
        translate_english=bool(args.translate),
    )
    if not prepared.words:
        print("No valid search terms after preprocessing. Aborting.")
        return
    search_words  = prepared.words
    word_labels   = prepared.labels
    origins_map   = prepared.origins
    if args.translate and not args.no_progress:
        print()

    engine = ELSEngine(
        min_skip=args.min_skip,
        max_skip=args.max_skip,
        texts_dir=args.texts_dir,
        data_dir=args.data_dir,
        show_load_progress=not args.no_progress,
        validate=args.validate,
        threads=args.threads if args.threads is not None else 1,
        long_skip=args.long_skip,
    )

    book_filter = _resolve_book_filter(args.books or None)

    # Dynamic skip scaling: widen the search range for long words
    if args.auto_scale:
        corpus_bytes = engine.corpus_bytes_for(book_filter)
        eff_skip = _effective_max_skip(
            search_words, args.max_skip, len(corpus_bytes), scale_to=args.scale_to
        )
        if eff_skip != args.max_skip:
            print(
                f"Auto-scale: max-skip raised {args.max_skip:,} → {eff_skip:,} "
                f"(word length > 5 in a {len(corpus_bytes):,}-letter corpus, scale-to={args.scale_to:,})"
            )
            engine.max_skip = eff_skip
            args.max_skip   = eff_skip

    # Resolve thread count after skip range is final (auto-scale may have widened it).
    # A user-supplied --threads value takes precedence; otherwise use workload heuristic.
    if args.threads is None:
        args.threads = _auto_threads(len(search_words), args.max_skip - args.min_skip)
        engine.threads = args.threads

    print(f"Searching for: {search_words}  |  skip range: ±[{args.min_skip}, {args.max_skip}]"
          + (f"  |  books: {book_filter}" if book_filter else "") + "\n")

    matches = engine.search(
        search_words,
        books=book_filter,
        show_progress=not args.no_progress,
        threads=args.threads,
    )

    cpu_eff = engine.last_cpu_secs / engine.last_wall_secs if engine.last_wall_secs > 0 else 0.0
    print(
        f"Timing summary: Wall-clock {engine.last_wall_secs:.3f}s  |  "
        f"CPU time {engine.last_cpu_secs:.3f}s  |  "
        f"Threads {engine.last_threads}  |  CPU/Wall {cpu_eff:.2f}x"
    )

    if not matches:
        print("No matches found.")
        return

    # Apply HeBERT score threshold when validation was requested
    if args.validate and args.min_score > 0.0:
        before = len(matches)
        matches = [m for m in matches if m.hebert_score >= args.min_score]
        print(f"Score filter (>= {args.min_score}): kept {len(matches)}/{before} matches")
        if not matches:
            print("No matches survived the score filter.")
            return

    # Run Monte Carlo baseline and enrich matches with Z-scores
    if args.baseline:
        print(f"\nRunning Monte Carlo baseline ({args.baseline_trials} trials) …")
        book_filter = _resolve_book_filter(args.books if args.books else None)
        sem_sample = args.sem_sample if (args.validate and args.baseline) else 0
        baseline = _stats.run_monte_carlo(
            engine.corpus_bytes_for(book_filter),
            search_words,
            args.min_skip,
            args.max_skip,
            n_trials=args.baseline_trials,
            show_progress=not args.no_progress,
            score_sample_size=sem_sample,
            long_skip=args.long_skip,
        )
        matches = _stats.apply_significance(
            matches,
            baseline,
            real_hit_count=len(matches),
            z_threshold=args.z_threshold,
        )
        sig_count = sum(1 for m in matches if m.is_significant)
        sem_info = (
            f"  |  semantic μ={baseline.score_mean:.4f} σ={baseline.score_std:.6f} "
            f"(N={sem_sample} samples)"
            if baseline.score_std > 0.0
            else ""
        )
        print(
            f"Baseline: μ={baseline.hit_count_mean:.1f} hits, "
            f"σ={baseline.hit_count_std:.2f}  over {baseline.n_trials} trials\n"
            f"Real hits: {len(matches)}   Z = {matches[0].z_score:.2f}   "
            f"Significant: {sig_count}/{len(matches)} matches"
            + sem_info
        )

    matches.sort(key=lambda m: (abs(m.skip), m.book, m.start))
    display = matches[: args.top] if args.top > 0 else matches

    print(
        f"\nFound {len(matches)} match(es)"
        + (f" (showing top {args.top})" if args.top and args.top < len(matches) else "")
        + ":\n"
    )

    score_col = args.validate
    stat_col  = args.baseline
    sem_col   = args.baseline and args.validate and any(m.semantic_z_score != 0.0 for m in display)
    label_col = args.translate or any(
        label != word for word, label in zip(search_words, word_labels)
    )  # show Label column when any term was translated, converted, or dated
    header = (
        f"{'Book':<14}  {'Verse':<22}  {'Word':<12}  "
        f"{'Sequence':<12}  {'Skip':>6}  {'Start':>8}"
        + ("  Score" if score_col else "")
        + ("     Z   Sig" if stat_col else "")
        + ("   SemZ" if sem_col else "")
        + (f"  {'Label':<28}" if label_col else "")
    )
    print(header)
    print("-" * len(header))
    for m in display:
        line = (
            f"{m.book:<14}  {m.verse:<22}  {m.word:<12}  "
            f"{m.sequence:<12}  {m.skip:>6}  {m.start:>8}"
        )
        if score_col:
            line += f"  {m.hebert_score:.4f}"
        if stat_col:
            sig_marker = " ★" if m.is_significant else "  "
            line += f"  {m.z_score:>6.2f}{sig_marker}"
        if sem_col:
            line += f"  {m.semantic_z_score:>6.2f}"
        if label_col:
            lbl = origins_map.get(m.word, m.word)
            line += f"  {lbl:<28}"
        print(line)

    if args.output:
        out_path = _reporter.export_results(
            matches,
            format=args.output_format,
            filename=args.output,
        )
        print(f"\nExported {len(matches)} match(es) to {out_path}")

    if args.archive:
        if not (args.validate and args.baseline):
            print(
                "\n[archive] Skipped — --archive requires both --validate and --baseline "
                "to be active (HeBERT embeddings and Z-scores must both be present)."
            )
        else:
            n_archived = _archiver.archive_matches(matches)
            stats_info = _archiver.db_stats()
            print(
                f"\n[archive] Stored {n_archived} significant match(es) → "
                f"{stats_info['db_dir']}  "
                f"(collection '{stats_info['collection']}', total {stats_info['count']} entries)"
            )

    if args.view_grid:
        if len(matches) < 2:
            print("\n--view-grid requires at least 2 matches; skipping grid render.")
            return
        width = args.grid_width if args.grid_width > 0 else 0
        if width > 0:
            table = _gridder.find_table(matches, min_width=width, max_width=width)
        else:
            table = _gridder.find_table(matches)

        # Build the grid from the corpus segment spanning all matches
        all_starts = [m.start for m in matches]
        grid_start = max(0, min(all_starts) - table.width)
        grid_end   = max(all_starts) + table.width * 2

        book_filter_for_grid = list({m.book for m in matches})
        corpus_bytes = engine.corpus_bytes_for(book_filter_for_grid)
        # Remap grid indices: corpus_bytes_for returns a slice; find offset of first match
        # Use full corpus bytes so start indices align with Match.start values
        full_bytes = engine.corpus_bytes
        grid = _gridder.get_grid(full_bytes, grid_start, grid_end, table.width)

        html_path = _reporter.render_grid_to_html(
            grid,
            matches,
            grid_start_index=grid_start,
            filename=args.grid_output,
        )
        print(f"\nGrid written to {html_path}  (width={table.width}, score={table.cluster_score:.2f})")
        import webbrowser
        try:
            webbrowser.open(html_path.as_uri())
        except Exception:
            pass  # silently skip if no browser available


if __name__ == "__main__":
    main()
