"""
bot/manager.py — CodeSearch-Ultra Platform Orchestrator.

The ``PlatformManager`` drives the full "All-of-it" pipeline:

    1. News Trigger ingestion  — query News API for daily geopolitical keywords
                                  and convert them to Hebrew ELS search terms.
    2. Automated Search        — run UltraSearchEngine with Chronos Anchor
                                  cross-reference enabled.
    3. Analysis                — build / update the VerseYearNetwork graph.
    4. Visualisation artefacts — export adjacency CSV + GraphML.
    5. GitHub commit           — generate a cold-intelligence Markdown briefing
                                  and commit it to docs/decadal_horizon.md.

CLI usage
---------
    uv run bot/manager.py --words משיח --commit
    uv run bot/manager.py --words Jerusalem Hormuz --translate --commit --dry-run

Environment variables
---------------------
    NEWS_API_KEY    — newsapi.org key (optional; stub returns canned terms if absent)
    GH_TOKEN        — GitHub personal-access token (optional; used only if pushing
                      to a remote.  Local commits require only git being configured.)
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import threading
from pathlib import Path
from typing import Sequence

# Force UTF-8 on Windows consoles so Hebrew print() calls don't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── resolve paths ─────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "backend"
_DOCS    = _ROOT / "docs"

# Insert BACKEND first, then ROOT — so ROOT ends up at index 0 and the
# engine/ package (project root) takes precedence over backend/engine.py.
# backend/ must still be present so that backend's own bare-name imports
# (data_loader, validator, etc.) resolve correctly.
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Chronos Anchors ───────────────────────────────────────────────────────────
CHRONOS_ANCHORS: tuple[str, ...] = ("תשפו", "תשצ", "תשצו")

# Geopolitical seed keywords (Hebrew ELS equivalents patched in by _translate stub)
_GEO_KEYWORDS: list[str] = [
    "ירושלים",   # Jerusalem
    "הורמוז",    # Hormuz
    "חמש_אייר",  # 5 Iyar
    "נשיא",      # President
    "ברית",      # Covenant
    "שלום",      # Peace
]


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL INGESTOR
# ══════════════════════════════════════════════════════════════════════════════

#  Critical threshold: hits above this score trigger an immediate CRITICAL
#  SIGNAL commit tagged separately from the regular daily briefing.
_CRITICAL_THRESHOLD: float = 0.50

# NewsAPI topics for the signal feed
_SIGNAL_TOPICS: list[str] = ["Israel", "Iran", "Jerusalem", "Tesla IPO"]

# Polling interval (seconds).  4 hours by default; overridable for testing.
_POLL_INTERVAL: int = 4 * 60 * 60


class SignalIngestor:
    """
    Continuously polls NewsAPI for top headlines on ``_SIGNAL_TOPICS``,
    translates extracted keywords to Hebrew via the Translation Bridge,
    and fires an ELS search on each cycle.

    If any hit exceeds ``_CRITICAL_THRESHOLD`` (0.50 consensus) the
    commit message is tagged **CRITICAL SIGNAL** and the commit is
    executed regardless of the ``dry_run`` flag.

    Parameters
    ----------
    api_key : str | None
        newsapi.org key.  Falls back to ``NEWS_API_KEY`` env var.
    max_skip : int
        ELS skip range.
    poll_interval : int
        Seconds between fetch cycles (default 14 400 = 4 h).
    repo_path : Path
        Git repository root for commits.
    dry_run : bool
        When True regular commits are skipped, but CRITICAL SIGNAL commits
        are still executed (they override the dry-run flag by design).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_skip: int = 1_000,
        poll_interval: int = _POLL_INTERVAL,
        repo_path: str | Path = _ROOT,
        dry_run: bool = False,
    ) -> None:
        self._api_key       = api_key or os.environ.get("NEWS_API_KEY")
        self._max_skip      = max_skip
        self._poll_interval = poll_interval
        self._repo_path     = Path(repo_path)
        self._dry_run       = dry_run
        self._stop_event    = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────

    def start_background(self) -> threading.Thread:
        """
        Launch the ingestion loop in a daemon thread.
        Call ``stop()`` to terminate.
        """
        t = threading.Thread(target=self._loop, daemon=True, name="SignalIngestor")
        t.start()
        print(f"[SignalIngestor] Started background thread (interval={self._poll_interval}s).")
        return t

    def stop(self) -> None:
        """Signal the background loop to exit after its current sleep."""
        self._stop_event.set()

    def run_once(self) -> dict:
        """
        Execute a single ingestion cycle synchronously.

        Returns
        -------
        dict with keys: keywords, results, critical_hits, commit_sha
        """
        return self._cycle()

    # ── Internal ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._cycle()
            except Exception as exc:  # noqa: BLE001
                print(f"[SignalIngestor] Cycle error: {exc}")
            self._stop_event.wait(timeout=self._poll_interval)

    def _cycle(self) -> dict:
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[SignalIngestor] Cycle start {ts}")

        # Step 1: fetch keywords
        keywords = self._fetch_signal_keywords()
        print(f"[SignalIngestor] Keywords: {keywords[:8]}")
        if not keywords:
            return {"keywords": [], "results": [], "critical_hits": [], "commit_sha": None}

        # Step 2: search
        from engine.search import UltraSearchEngine
        eng = UltraSearchEngine(max_skip=self._max_skip)
        results = eng.run(
            keywords,
            validate=True,
            dual_score=True,
            check_anchors=True,
        )
        print(f"[SignalIngestor] {len(results)} hit(s).")

        # Step 3: triage critical hits (consensus > 0.50)
        critical_hits = [
            (m, cs) for m, cs in results if cs.consensus > _CRITICAL_THRESHOLD
        ]

        sha: str | None = None
        if critical_hits:
            print(f"[SignalIngestor] *** CRITICAL SIGNAL *** {len(critical_hits)} hit(s) above 0.50")
            top = critical_hits[:3]
            top3 = [
                {
                    "word":             m.word,
                    "verse":            m.verse,
                    "book":             m.book,
                    "skip":             m.skip,
                    "consensus":        cs.consensus,
                    "is_decadal_anchor": cs.is_decadal_anchor,
                }
                for m, cs in top
            ]
            # CRITICAL SIGNAL commits always go through (override dry_run)
            sha = self._critical_commit(top3, ts=ts)
        elif not self._dry_run and results:
            # Regular cycle commit (top-3)
            top3 = [
                {
                    "word":             m.word,
                    "verse":            m.verse,
                    "book":             m.book,
                    "skip":             m.skip,
                    "consensus":        cs.consensus,
                    "is_decadal_anchor": cs.is_decadal_anchor,
                }
                for m, cs in results[:3]
            ]
            sha = sync_to_github(top3, dry_run=False)

        return {
            "keywords":     keywords,
            "results":      results,
            "critical_hits": critical_hits,
            "commit_sha":   sha,
        }

    def _fetch_signal_keywords(self) -> list[str]:
        """Fetch top headlines for ``_SIGNAL_TOPICS`` and translate to Hebrew."""
        raw_english: list[str] = []
        key = self._api_key

        if key:
            try:
                import requests  # type: ignore
                for topic in _SIGNAL_TOPICS:
                    url = (
                        "https://newsapi.org/v2/everything"
                        f"?q={topic}&language=en&sortBy=publishedAt&pageSize=5"
                    )
                    resp = requests.get(url, headers={"X-Api-Key": key}, timeout=10)
                    resp.raise_for_status()
                    for art in resp.json().get("articles", []):
                        title = art.get("title") or ""
                        raw_english.extend(w.strip(".,!?\"'") for w in title.split()
                                           if len(w.strip(".,!?\"'")) >= 3)
            except Exception as exc:  # noqa: BLE001
                print(f"[SignalIngestor] NewsAPI error: {exc}")

        if not raw_english:
            # Fall back: use the English signal topics as seeds
            raw_english = list(_SIGNAL_TOPICS)

        # Translate via cache + deep_translator and deduplicate
        seen: set[str] = set()
        hebrew_terms: list[str] = []
        for eng_word in raw_english:
            heb = _translate(eng_word)
            if heb and heb not in seen:
                seen.add(heb)
                hebrew_terms.append(heb)

        return hebrew_terms[:20]

    def _critical_commit(self, top3: list[dict], *, ts: str) -> str:
        """Write and commit a CRITICAL SIGNAL briefing (ignores dry_run)."""
        _DOCS.mkdir(parents=True, exist_ok=True)
        date_str = ts[:10]
        he_year  = _estimate_hebrew_year(int(date_str[:4]))
        briefing = _build_briefing(
            top3, date_str=date_str, ts_str=ts, he_year=he_year, critical=True
        )
        target = _DOCS / "decadal_horizon.md"
        target.write_text(briefing, encoding="utf-8")
        try:
            import git  # type: ignore
            repo   = git.Repo(self._repo_path)
            repo.index.add([str(target.relative_to(self._repo_path))])
            msg    = (
                f"alert(CRITICAL SIGNAL): {len(top3)} hit(s) >0.50 consensus "
                f"[{ts}] -- auto-commit by SignalIngestor"
            )
            commit = repo.index.commit(msg)
            sha    = commit.hexsha[:8]
            print(f"[SignalIngestor] CRITICAL commit → {sha}")
            return sha
        except Exception as exc:  # noqa: BLE001
            print(f"[SignalIngestor] CRITICAL commit failed: {exc}")
            return f"[critical-commit-failed: {exc}]"


def _translate(english: str) -> str:
    """Translate an English string to Hebrew, cache-first."""
    key = f"en:iw:{english.strip().lower()}"
    _cache_path = _ROOT / "data" / "translation_cache.json"
    import json
    cache: dict = {}
    if _cache_path.exists():
        try:
            cache = json.loads(_cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if key in cache:
        return cache[key]
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        result = GoogleTranslator(source="en", target="iw").translate(english.strip())
        if result:
            cache[key] = result
            _cache_path.parent.mkdir(parents=True, exist_ok=True)
            _cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
    except Exception:
        pass
    return english


# ══════════════════════════════════════════════════════════════════════════════
#  NEWS API STUB
# ══════════════════════════════════════════════════════════════════════════════

def fetch_news_keywords(api_key: str | None = None) -> list[str]:
    """
    Query newsapi.org for today's top geopolitical headlines and extract
    Hebrew ELS search candidates.

    If ``api_key`` is None or the request fails, fall back to the static
    ``_GEO_KEYWORDS`` seed list so the pipeline never halts.

    Parameters
    ----------
    api_key : str | None
        newsapi.org key.  Reads ``NEWS_API_KEY`` env variable if omitted.

    Returns
    -------
    list[str]
        Hebrew search terms (may include transliterated proper nouns).
    """
    key = api_key or os.environ.get("NEWS_API_KEY")
    if not key:
        return list(_GEO_KEYWORDS)

    try:
        import requests  # type: ignore

        # Geopolitical query for today's intelligence feed
        query_terms = "Jerusalem OR Hormuz OR Israel OR Iran OR Gaza OR Saudi OR Trump OR Netanyahu"
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={query_terms}"
            "&language=he"
            "&sortBy=publishedAt"
            "&pageSize=20"
        )
        resp = requests.get(url, headers={"X-Api-Key": key}, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

        # Extract unique title words (Hebrew, 2+ characters)
        seen: set[str] = set()
        candidates: list[str] = list(_GEO_KEYWORDS)
        for art in articles:
            title = art.get("title") or ""
            for token in title.split():
                token = token.strip(".,!?\"'")
                if len(token) >= 2 and token not in seen:
                    seen.add(token)
                    candidates.append(token)

        return candidates[:30]

    except Exception as exc:  # noqa: BLE001
        print(f"[NewsAPI] Request failed ({exc}); using static keyword list.")
        return list(_GEO_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
#  GITHUB COMMIT
# ══════════════════════════════════════════════════════════════════════════════

def sync_to_github(
    top3: list[dict],
    *,
    repo_path: str | Path = _ROOT,
    dry_run: bool = False,
) -> str:
    """
    Generate a cold-intelligence Markdown briefing for today's top-3 semantic
    anomalies and commit it to ``docs/decadal_horizon.md``.

    Parameters
    ----------
    top3 : list[dict]
        Up to 3 dicts with keys: word, verse, book, skip,
        consensus, is_decadal_anchor.
    repo_path : Path
        Local git repository root (default: project root).
    dry_run : bool
        When True, write the file but do NOT execute the git commit.

    Returns
    -------
    str
        The commit SHA (or "[dry-run]" if dry_run=True).
    """
    repo_path = Path(repo_path)
    _DOCS.mkdir(parents=True, exist_ok=True)
    target = _DOCS / "decadal_horizon.md"

    now       = datetime.datetime.utcnow()
    date_str  = now.strftime("%Y-%m-%d")
    ts_str    = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    he_year   = _estimate_hebrew_year(now.year)

    briefing = _build_briefing(top3, date_str=date_str, ts_str=ts_str, he_year=he_year)
    target.write_text(briefing, encoding="utf-8")

    if dry_run:
        print(f"[sync_to_github] Dry-run: wrote {target} (no commit)")
        return "[dry-run]"

    try:
        import git  # type: ignore
        repo   = git.Repo(repo_path)
        repo.index.add([str(target.relative_to(repo_path))])
        msg    = f"chore(intel): decadal briefing {date_str} — top-3 anomalies [{ts_str}]"
        commit = repo.index.commit(msg)
        sha    = commit.hexsha[:8]
        print(f"[sync_to_github] Committed {target.name} → {sha}")
        return sha
    except Exception as exc:  # noqa: BLE001
        print(f"[sync_to_github] Git commit failed ({exc}); file written at {target}")
        return f"[commit-failed: {exc}]"


def _estimate_hebrew_year(gregorian_year: int) -> str:
    """
    Return the dominant Hebrew year for a Gregorian year as a string.
    (Accurate within ±1 for display purposes only.)
    """
    # Jewish year begins in autumn; add 3760 for Tishri-year approximation.
    return str(gregorian_year + 3760)


def _build_briefing(
    top3: list[dict],
    *,
    date_str: str,
    ts_str: str,
    he_year: str,
    critical: bool = False,
) -> str:
    """
    Produce a cold, forensic intelligence briefing in Markdown.

    Systemic Decadal Anchors (תשפו / תשצ / תשצו) are flagged prominently.
    When *critical* is True, the header is marked CRITICAL SIGNAL.
    """
    lines: list[str] = [
        "# Decadal Horizon — Semantic Anomaly Report",
        "",
        f"**Classification:** {'*** CRITICAL SIGNAL ***' if critical else 'INTERNAL INTELLIGENCE SUMMARY'}  ",
        f"**Date:** {date_str}  ",
        f"**UTC Timestamp:** {ts_str}  ",
        f"**Hebrew Year (approx.):** {he_year}  ",
        f"**Chronos Anchors in scope:** תשפו (5786/2026) · תשצ (5790/2030) · תשצו (5796/2036)  ",
        "",
        "---",
        "",
        "## Top-3 Semantic Anomalies",
        "",
    ]

    if not top3:
        lines.append("_No anomalies above threshold recorded in this cycle._")
    else:
        for rank, entry in enumerate(top3[:3], start=1):
            word      = entry.get("word", "—")
            verse     = entry.get("verse", "—")
            book      = entry.get("book", "—")
            skip      = entry.get("skip", "—")
            consensus = entry.get("consensus", 0.0)
            anchor    = entry.get("is_decadal_anchor", False)

            anchor_flag = "  **⚑ SYSTEMIC DECADAL ANCHOR**" if anchor else ""

            lines += [
                f"### Rank {rank}{anchor_flag}",
                "",
                f"| Field | Value |",
                f"|-------|-------|",
                f"| **ELS Term** | `{word}` |",
                f"| **Verse** | {verse} ({book}) |",
                f"| **Skip** | {skip} |",
                f"| **Consensus Score** | `{consensus:.4f}` |",
                f"| **Decadal Anchor** | {'YES — Systemic Decadal Anchor' if anchor else 'No'} |",
                "",
            ]

    lines += [
        "---",
        "",
        "## Methodology",
        "",
        "Scores represent the arithmetic mean of HeBERT (avichr/heBERT) and "
        "AlephBERT (onlplab/alephbert-base) cosine similarities between the "
        "ELS match term and its immediate three-verse context window.  "
        "Significance threshold: **0.30**.  "
        "Systemic Decadal Anchor flag is raised when co-location with "
        "תשפו, תשצ, or תשצו is confirmed within the same skip band.",
        "",
        f"_Generated by CodeSearch-Ultra · {ts_str}_",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  PLATFORM MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class PlatformManager:
    """
    Orchestrates the full CodeSearch-Ultra pipeline.

    Usage (programmatic)
    --------------------
    >>> mgr = PlatformManager()
    >>> mgr.run_all(words=["משיח", "ירושלים"], commit=True)

    Usage (CLI)
    -----------
    See ``__main__`` block at module bottom.
    """

    def __init__(
        self,
        *,
        max_skip: int = 1_000,
        news_api_key: str | None = None,
        dry_run: bool = False,
    ) -> None:
        self._max_skip     = max_skip
        self._news_api_key = news_api_key or os.environ.get("NEWS_API_KEY")
        self._dry_run      = dry_run

    # ── Main entry point ──────────────────────────────────────────────────────

    def run_all(
        self,
        words: Sequence[str] | None = None,
        *,
        books: list[str] | None = None,
        use_news_triggers: bool = False,
        commit: bool = False,
    ) -> dict:
        """
        Execute the full automated pipeline.

        Steps
        -----
        1. Resolve search terms (from *words* and/or News API triggers).
        2. Append Chronos Anchors.
        3. Run UltraSearchEngine.
        4. Build / update VerseYearNetwork.
        5. Export adjacency artefacts.
        6. Optionally commit briefing to docs/decadal_horizon.md.

        Returns
        -------
        dict with keys: results, network, commit_sha, top3
        """
        # ── Step 1: Resolve terms ─────────────────────────────────────────────
        terms: list[str] = list(words) if words else []

        if use_news_triggers:
            news_terms = fetch_news_keywords(self._news_api_key)
            print(f"[PlatformManager] News API returned {len(news_terms)} terms.")
            terms.extend(news_terms)

        # Deduplicate preserving order
        seen: set[str] = set()
        unique_terms: list[str] = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                unique_terms.append(t)

        if not unique_terms:
            print("[PlatformManager] No search terms after deduplication.  Aborting.")
            return {"results": [], "network": None, "commit_sha": None, "top3": []}

        # ── Step 2–3: Search ──────────────────────────────────────────────────
        print(f"[PlatformManager] Running search: {unique_terms[:6]} ... "
              f"(max_skip={self._max_skip})")

        from engine.search import UltraSearchEngine
        eng = UltraSearchEngine(max_skip=self._max_skip)
        results = eng.run(
            words=unique_terms,
            books=books,
            validate=True,
            dual_score=True,
            check_anchors=True,
        )
        print(f"[PlatformManager] {len(results)} significant hit(s).")

        # ── Step 4: Network graph ─────────────────────────────────────────────
        from analysis.network import VerseYearNetwork
        net = VerseYearNetwork()
        net.ingest(results)
        print(f"[PlatformManager] Network: {net.node_count} nodes, {net.edge_count} edges.")

        # ── Step 5: Export artefacts ──────────────────────────────────────────
        today = datetime.date.today().strftime("%Y%m%d")
        _DOCS.mkdir(parents=True, exist_ok=True)
        adj_path = _DOCS / f"network_{today}.csv"
        gml_path = _DOCS / f"network_{today}.graphml"
        net.export_adjacency(adj_path)
        net.export_graphml(gml_path)
        print(f"[PlatformManager] Adjacency → {adj_path}")
        print(f"[PlatformManager] GraphML   → {gml_path}")

        # ── Top-3 extraction ──────────────────────────────────────────────────
        top3_raw = [
            {
                "word":             m.word,
                "verse":            m.verse,
                "book":             m.book,
                "skip":             m.skip,
                "consensus":        cs.consensus,
                "is_decadal_anchor": cs.is_decadal_anchor,
            }
            for m, cs in results[:3]
        ]

        # ── Step 6: Git commit ────────────────────────────────────────────────
        sha: str | None = None
        if commit:
            sha = sync_to_github(top3_raw, dry_run=self._dry_run)

        return {
            "results":    results,
            "network":    net,
            "commit_sha": sha,
            "top3":       top3_raw,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manager",
        description="CodeSearch-Ultra Platform Orchestrator",
    )
    p.add_argument("--words",    nargs="*", default=None,
                   help="Hebrew search terms.")
    p.add_argument("--books",    nargs="*", default=None,
                   help="Book filter (e.g. Torah Genesis Isaiah).")
    p.add_argument("--max-skip", type=int, default=1_000,
                   dest="max_skip")
    p.add_argument("--news",     action="store_true",
                   help="Fetch live geopolitical keywords from News API.")
    p.add_argument("--ingest",   action="store_true",
                   help="Run one SignalIngestor cycle (fetch, translate, search, commit if critical).")
    p.add_argument("--commit",   action="store_true",
                   help="Commit the daily briefing to docs/decadal_horizon.md.")
    p.add_argument("--dry-run",  action="store_true", dest="dry_run",
                   help="Write briefing file but skip git commit.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.ingest:
        ingestor = SignalIngestor(
            max_skip=args.max_skip,
            dry_run=args.dry_run,
        )
        result = ingestor.run_once()
        crits = result.get("critical_hits", [])
        if crits:
            print(f"*** CRITICAL SIGNAL: {len(crits)} hit(s) above 0.50 consensus ***")
        sha = result.get("commit_sha")
        if sha:
            print(f"[Git] Committed: {sha}")
        return

    mgr  = PlatformManager(
        max_skip=args.max_skip,
        dry_run=args.dry_run,
    )
    result = mgr.run_all(
        words=args.words,
        books=args.books,
        use_news_triggers=args.news,
        commit=args.commit,
    )
    top3 = result.get("top3", [])
    print("\n── Top-3 Semantic Anomalies ──────────────────────────────────")
    for i, entry in enumerate(top3, 1):
        anchor = " ★ SYSTEMIC DECADAL ANCHOR" if entry.get("is_decadal_anchor") else ""
        print(f"  {i}. {entry['word']}  |  {entry['verse']}  |  "
              f"skip={entry['skip']}  |  consensus={entry['consensus']:.4f}{anchor}")
    sha = result.get("commit_sha")
    if sha:
        print(f"\n[Git] Committed: {sha}")


if __name__ == "__main__":
    main()
