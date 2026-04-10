"""
Project-Manager.py — CodeSearch Orchestration Engine
=====================================================
Automates the full ELS research pipeline:
  1. Run the search engine (main.py via uv)
  2. Analyse the output CSV for verse locks and top anchors
  3. Append findings to the intelligence report
  4. Commit everything to Git

Usage
-----
  uv run Project-Manager.py --words "word1" "word2" --output run_68_label
  uv run Project-Manager.py --words "קושנר" "אייר" --books Tanakh --scale-to 100000 --output run_68_label --report docs/decadal_horizon_report.md
  uv run Project-Manager.py --words "Veto" "China" --translate --output run_69_label

All extra flags after --words / --books / --output / --report / --no-commit
are forwarded verbatim to main.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = PROJECT_ROOT / "decadal_horizon_report.md"
LOCK_SCORE_THRESHOLD = 0.30   # minimum HeBERT score for a hit to count
MIN_TOKENS_FOR_LOCK = 2       # minimum distinct tokens in the same verse
TOP_N_STANDALONE = 3          # top-N entries reported when no lock is found


# ── 1. Search wrapper ──────────────────────────────────────────────────────────

def run_search(words: list[str], extra_flags: list[str], output_csv: str) -> Path:
    """
    Invoke `uv run main.py` with the supplied words and extra CLI flags.
    --archive, --validate, and --books Tanakh NT are always injected
    (unless the caller already supplied --books in extra_flags).
    Returns the resolved path to the output CSV.
    """
    # Inject --books Tanakh NT only when the caller hasn't overridden it
    books_flags: list[str] = [] if "--books" in extra_flags else ["--books", "Tanakh", "NT"]

    cmd: list[str] = [
        "uv", "run", "main.py",
        "--words", *words,
        "--archive",
        "--validate",
        *books_flags,
        "--output", output_csv,
        *extra_flags,
    ]
    print(f"\n[PM] Running search engine:\n    {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        sys.exit(f"[PM] main.py exited with code {result.returncode}. Aborting.")

    # main.py appends .csv if not already present
    csv_path = PROJECT_ROOT / (output_csv if output_csv.endswith(".csv") else f"{output_csv}.csv")
    if not csv_path.exists():
        sys.exit(f"[PM] Expected output CSV not found: {csv_path}")
    return csv_path


# ── 2. CSV analysis ────────────────────────────────────────────────────────────

def _book_from_verse(verse: str) -> str:
    """Return the book name from a 'Book Chapter:Verse' string."""
    parts = verse.rsplit(" ", 1)
    return parts[0] if len(parts) == 2 else verse


def analyse_csv(csv_path: Path) -> dict:
    """
    Scan the output CSV and return a findings dict with:
      - locks: list of verse-lock records (verse, tokens, max_score)
      - top_entries: top-N entries by hebert_score across all tokens
      - total_rows: int
      - unique_tokens: list[str]
    """
    df = pl.read_csv(csv_path)

    total_rows = df.height
    unique_tokens: list[str] = sorted(df["word"].unique().to_list())

    # ── Verse locks: verses where 2+ distinct tokens appear above threshold ──
    qualified = df.filter(pl.col("hebert_score") >= LOCK_SCORE_THRESHOLD)

    lock_records = (
        qualified
        .group_by("verse")
        .agg([
            pl.col("word").unique().alias("tokens"),
            pl.col("hebert_score").max().alias("max_score"),
            pl.col("semantic_z_score").min().alias("min_semz"),
            pl.col("skip").first().alias("skip"),
        ])
        .filter(pl.col("tokens").list.len() >= MIN_TOKENS_FOR_LOCK)
        .sort("max_score", descending=True)
    )

    locks = []
    for row in lock_records.to_dicts():
        locks.append({
            "verse": row["verse"],
            "book": _book_from_verse(row["verse"]),
            "tokens": sorted(row["tokens"]),
            "max_score": round(row["max_score"], 4),
            "min_semz": round(row["min_semz"], 2),
        })

    # ── Top-N entries overall ──────────────────────────────────────────────────
    top_entries = (
        df
        .sort("hebert_score", descending=True)
        .unique(subset=["verse", "word"], keep="first")
        .head(TOP_N_STANDALONE)
        .select(["word", "verse", "skip", "hebert_score", "semantic_z_score"])
        .to_dicts()
    )
    for e in top_entries:
        e["hebert_score"] = round(e["hebert_score"], 4)
        e["semantic_z_score"] = round(e["semantic_z_score"], 2)

    return {
        "locks": locks,
        "top_entries": top_entries,
        "total_rows": total_rows,
        "unique_tokens": unique_tokens,
    }


# ── 3. Report integration ──────────────────────────────────────────────────────

def _format_report_block(run_label: str, csv_path: Path, findings: dict) -> str:
    """Render a Markdown section to append to the intelligence report."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    tokens_str = " · ".join(findings["unique_tokens"])
    lines: list[str] = [
        "",
        "---",
        "",
        f"## Run: {run_label}",
        "",
        f"> **Generated:** {now}  ",
        f"> **Source CSV:** `{csv_path.name}`  ",
        f"> **Total entries:** {findings['total_rows']:,}  ",
        f"> **Tokens:** {tokens_str}",
        "",
    ]

    locks = findings["locks"]
    top_entries = findings["top_entries"]

    if locks:
        lines += [
            f"### Verse Locks ({len(locks)} found)",
            "",
            "| Verse | Tokens | Max HeBERT | Min SemZ |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for lock in locks:
            tok_str = " + ".join(f"**{t}**" for t in lock["tokens"])
            lines.append(
                f"| {lock['verse']} | {tok_str} | **{lock['max_score']}** | {lock['min_semz']} |"
            )
        lines.append("")

        # Narrative for the top lock
        top = locks[0]
        tok_str = " and ".join(f"**{t}**" for t in top["tokens"])
        lines += [
            f"**Primary lock:** {tok_str} co-anchor at **{top['verse']}** "
            f"(H = {top['max_score']}, SemZ = {top['min_semz']}).",
            "",
        ]
    else:
        lines += [
            "### No verse locks above threshold",
            "",
            f"No verses contained {MIN_TOKENS_FOR_LOCK}+ distinct tokens "
            f"with HeBERT ≥ {LOCK_SCORE_THRESHOLD}. "
            "Top standalone entries recorded below.",
            "",
        ]

    lines += [
        "### Top Anchors",
        "",
        "| Token | Verse | Skip | HeBERT | SemZ |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for e in top_entries:
        lines.append(
            f"| **{e['word']}** | {e['verse']} | {e['skip']} "
            f"| **{e['hebert_score']}** | {e['semantic_z_score']} |"
        )
    lines.append("")

    return "\n".join(lines)


def append_to_report(report_path: Path, run_label: str, csv_path: Path, findings: dict) -> None:
    """Append the formatted findings block to the report file."""
    block = _format_report_block(run_label, csv_path, findings)
    with open(report_path, "a", encoding="utf-8") as fh:
        fh.write(block)
    print(f"[PM] Findings appended to: {report_path}")


# ── 4. Git automation ──────────────────────────────────────────────────────────

def _build_commit_message(run_label: str, findings: dict) -> str:
    """Build a dynamic, data-driven commit message."""
    locks = findings["locks"]
    if locks:
        top = locks[0]
        tokens_str = "+".join(top["tokens"])
        return (
            f"{run_label}: High-fidelity lock found in "
            f"{top['verse']} [{tokens_str}] score {top['max_score']}"
        )
    top_entries = findings["top_entries"]
    if top_entries:
        e = top_entries[0]
        return (
            f"{run_label}: Top anchor {e['word']} at "
            f"{e['verse']} score {e['hebert_score']}"
        )
    return f"{run_label}: Run complete — {findings['total_rows']} entries, no significant locks"


def git_commit(csv_path: Path, report_path: Path, commit_message: str) -> None:
    """Stage the CSV and report, then create a commit."""
    files_to_add = [str(csv_path), str(report_path)]

    add_result = subprocess.run(
        ["git", "add", *files_to_add],
        cwd=str(PROJECT_ROOT),
    )
    if add_result.returncode != 0:
        print("[PM] WARNING: git add failed — skipping commit.")
        return

    commit_result = subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=str(PROJECT_ROOT),
    )
    if commit_result.returncode != 0:
        print("[PM] WARNING: git commit failed (nothing staged, or other error).")
    else:
        print(f"[PM] Git commit: {commit_message}")


# ── Auto-naming helpers ───────────────────────────────────────────────────────

import re

def _get_next_run_id() -> int:
    """Scan existing run_NN_*.csv files and return the next integer run ID."""
    ids = []
    for p in PROJECT_ROOT.glob("run_*.csv"):
        m = re.match(r"run_(\d+)", p.name)
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else 68


def _slugify(words: list[str]) -> str:
    """Build a short snake_case slug from search terms (ASCII-safe)."""
    import unicodedata
    parts = []
    for w in words[:3]:  # cap slug length at 3 terms
        # drop non-ASCII (Hebrew), keep alphanumeric + space
        ascii_only = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
        cleaned = re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")
        if cleaned:
            parts.append(cleaned)
    return "_".join(parts) if parts else "scan"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Project-Manager: CodeSearch orchestration engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Positional args are treated as search words when --words is omitted.\n"
            "--output is auto-generated as run_NN_slug if omitted.\n\n"
            "Examples:\n"
            "  python Project-Manager.py \"5 Iyar\" Jerusalem \"Temple Mount\"\n"
            "  uv run Project-Manager.py --words קושנר אייר --books Tanakh "
            "--scale-to 100000 --baseline --output run_69_label"
        ),
    )
    parser.add_argument(
        "positional_words",
        nargs="*",
        metavar="WORD",
        help="Search terms as positional arguments (alternative to --words).",
    )
    parser.add_argument(
        "--words",
        nargs="+",
        metavar="WORD",
        help="Search terms (Hebrew or English). Forwarded to main.py --words.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="LABEL",
        help="Output CSV stem (auto-generated from run number + slug if omitted).",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        metavar="PATH",
        help=f"Path to the intelligence report to append findings to (default: {DEFAULT_REPORT}).",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Skip the Git commit step.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, extra_flags = parse_args()

    # Resolve word list: --words takes precedence over positional args
    words: list[str] = args.words if args.words else args.positional_words
    if not words:
        sys.exit("[PM] Error: supply search terms via positional args or --words.")

    # Auto-generate output name if not supplied
    output_label: str
    if args.output:
        output_label = args.output
    else:
        run_id = _get_next_run_id()
        slug = _slugify(words)
        output_label = f"run_{run_id}_{slug}"
        print(f"[PM] Auto-named output: {output_label}.csv")

    run_label = output_label.replace("_", " ").replace("-", " ").title()
    report_path = Path(args.report)

    if not report_path.exists():
        sys.exit(f"[PM] Report file not found: {report_path}\n"
                 "     Pass --report to specify a different path.")

    # ── Step 1: Run the search engine ─────────────────────────────────────────
    csv_path = run_search(
        words=words,
        extra_flags=extra_flags,
        output_csv=output_label,
    )

    # ── Step 2: Analyse the CSV ────────────────────────────────────────────────
    print(f"[PM] Analysing: {csv_path.name}")
    findings = analyse_csv(csv_path)

    print(f"[PM] Total entries : {findings['total_rows']:,}")
    print(f"[PM] Tokens found  : {', '.join(findings['unique_tokens'])}")
    print(f"[PM] Verse locks   : {len(findings['locks'])}")
    if findings["locks"]:
        for lock in findings["locks"][:5]:
            print(
                f"      → {lock['verse']} | "
                + "+".join(lock["tokens"])
                + f" | H={lock['max_score']}"
            )

    # ── Step 3: Append to report ───────────────────────────────────────────────
    append_to_report(report_path, run_label, csv_path, findings)

    # ── Step 4: Git commit ─────────────────────────────────────────────────────
    if not args.no_commit:
        commit_msg = _build_commit_message(run_label, findings)
        git_commit(csv_path, report_path, commit_msg)
    else:
        print("[PM] --no-commit: skipping Git step.")

    print("\n[PM] Pipeline complete.")


if __name__ == "__main__":
    main()
