"""
viz/dashboard.py — Streamlit intelligence dashboard for CodeSearch-Ultra.

Launch with:
    uv run streamlit run viz/dashboard.py

Panels
------
  1. Watchlist      — High-scoring hits (consensus >= 0.30) across all archived
                       runs, sourced from the ChromaDB collection.
  2. Live Search    — Interactive ELS search with dual-model scoring.
  3. ELS Grid       — 2D matrix view for any selected match.
  4. Network Map    — Force-directed verse-year graph (pyvis embedded as iframe).
  5. Chronos Audit  — Summary of Decadal Anchor cross-references.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── resolve backend on sys.path ───────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import streamlit as st
import polars as pl

# ── Translation cache ─────────────────────────────────────────────────────────
_CACHE_PATH = _ROOT / "data" / "translation_cache.json"


def _load_translation_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_translation_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _translate_to_hebrew(english: str) -> str:
    """Translate an English term to Hebrew using cache-first, then deep_translator."""
    key = f"en:iw:{english.strip().lower()}"
    cache = _load_translation_cache()
    if key in cache:
        return cache[key]
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        hebrew = GoogleTranslator(source="en", target="iw").translate(english.strip())
        if hebrew:
            cache[key] = hebrew
            _save_translation_cache(cache)
            return hebrew
    except Exception:
        pass
    return english  # fallback: return input unchanged


# ── platform imports (lazy to avoid hard crash at import time) ────────────────
@st.cache_resource
def _get_engine():
    # Force fresh load: drop any stale cached copy before importing.
    import importlib, sys
    for key in list(sys.modules):
        if key == "engine.search" or key == "engine":
            sys.modules.pop(key, None)
    from engine.search import UltraSearchEngine
    return UltraSearchEngine()


def _get_network():
    from analysis.network import VerseYearNetwork
    return VerseYearNetwork()


def _get_analyzer(net):
    from analysis.network import GraphAnalyzer
    return GraphAnalyzer(net)


def _paint(corpus_bytes, start, skip, word, english_label=""):
    from viz.grid_painter import paint_grid
    return paint_grid(corpus_bytes, start, skip, word, english_label=english_label)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CodeSearch-Ultra",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("CodeSearch-Ultra")
st.sidebar.caption("ELS Intelligence Platform — Cold, forensic mode.")
panel = st.sidebar.radio(
    "Panel",
    ["Watchlist", "Live Search", "ELS Grid", "Network Hub", "Chronos Audit"],
    index=0,
)

# ── Shared session state ──────────────────────────────────────────────────────
if "last_results" not in st.session_state:
    st.session_state["last_results"] = []
if "search_terms" not in st.session_state:
    st.session_state["search_terms"] = []        # list of {"hebrew": ..., "english": ...}
if "search_terms_snapshot" not in st.session_state:
    st.session_state["search_terms_snapshot"] = []


# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════
if panel == "Watchlist":
    st.header("Watchlist — High-Significance ELS Hits")
    st.caption("Source: ChromaDB persistent collection `bible_codes`. "
               "Threshold: consensus ≥ 0.30.")

    try:
        import chromadb  # type: ignore
        db_dir = _ROOT / "data" / "chroma"
        client = chromadb.PersistentClient(path=str(db_dir))
        col    = client.get_or_create_collection("bible_codes")

        # Paginate to avoid SQLite's ~32 766 variable limit when the
        # collection is large.  Fetch metadatas + documents in pages of 500.
        _PAGE  = 500
        rows: list[dict] = []
        offset = 0
        while True:
            res   = col.get(include=["metadatas", "documents"],
                            limit=_PAGE, offset=offset)
            metas = res.get("metadatas") or []
            docs  = res.get("documents") or []
            if not metas:
                break
            for meta, doc in zip(metas, docs):
                if meta and float(meta.get("hebert_score", 0)) >= 0.30:
                    rows.append({
                        "word":         meta.get("word", ""),
                        "verse":        meta.get("verse", ""),
                        "book":         meta.get("book", ""),
                        "skip":         meta.get("skip", 0),
                        "hebert_score": round(float(meta.get("hebert_score", 0)), 4),
                        "z_score":      round(float(meta.get("z_score", 0)), 3),
                        "significant":  meta.get("is_significant", False),
                        "sequence":     doc or "",
                    })
            if len(metas) < _PAGE:
                break
            offset += _PAGE

        if rows:
            df = pl.DataFrame(rows).sort("hebert_score", descending=True)
            st.dataframe(
                df.to_pandas(),
                use_container_width=True,
                height=500,
            )
            st.download_button(
                "Export Watchlist CSV",
                df.write_csv(),
                file_name="watchlist.csv",
                mime="text/csv",
            )
        else:
            st.info("No archived hits yet.  Run a search with --validate to populate.")

    except Exception as exc:  # noqa: BLE001
        st.error(f"ChromaDB unavailable: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif panel == "Live Search":
    # ── Translation Bridge (sidebar) ─────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("Translation Bridge")
        eng_input = st.text_input(
            "Input English Term",
            key="eng_input_term",
            placeholder="e.g. messiah, 2026",
        )
        col_add, col_clr = st.sidebar.columns(2)
        with col_add:
            if st.button("Add to Search", key="add_search_term"):
                raw = eng_input.strip()
                if raw:
                    heb = _translate_to_hebrew(raw)
                    entry = {"hebrew": heb, "english": raw}
                    if entry not in st.session_state["search_terms"]:
                        st.session_state["search_terms"].append(entry)
        with col_clr:
            if st.button("Clear", key="clear_search_terms"):
                st.session_state["search_terms"] = []

        terms = st.session_state["search_terms"]
        if terms:
            st.caption("Active terms:")
            for t in terms:
                st.markdown(f"**{t['hebrew']}** ({t['english']})")

    # ── Main panel ────────────────────────────────────────────────────────────
    st.header("Live ELS Search")

    # Search Term Box — Hebrew tokens with English meanings in parentheses
    terms = st.session_state["search_terms"]
    if terms:
        display = "  \u2003".join(f"{t['hebrew']} ({t['english']})" for t in terms)
        st.info(f"**Search Terms:** {display}")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        words_input = st.text_input(
            "Additional Hebrew terms (space-separated)",
            value="" if terms else "משיח",
            help="Append raw Hebrew terms to the translated terms above.",
        )
    with col2:
        max_skip = st.number_input("Max Skip", min_value=1, max_value=50_000, value=1_000, step=100)
    with col3:
        dual_score = st.checkbox("Dual-model scoring", value=True)

    books_options = [
        "All", "Torah", "Neviim", "Ketuvim", "Tanakh",
        "Genesis", "Exodus", "Isaiah", "Psalms", "Daniel",
    ]
    selected_books = st.multiselect("Books", books_options, default=["All"])
    books_arg = None if "All" in selected_books else [b for b in selected_books if b != "All"]

    if st.button("Run Search", type="primary"):
        words_from_state = [t["hebrew"] for t in terms]
        words_from_input = [w.strip() for w in words_input.replace(",", " ").split() if w.strip()]
        all_words = words_from_state + [w for w in words_from_input if w not in words_from_state]

        if not all_words:
            st.warning("Enter at least one search term via Translation Bridge or the text box.")
        else:
            progress_bar = st.progress(0.0, text="Initializing search...")
            results: list = []
            try:
                eng = _get_engine()
                eng._engine.max_skip = int(max_skip)
                eng._max_skip        = int(max_skip)

                # Prefer run_with_progress (streaming); fall back to run()
                # if the cached engine object pre-dates the method.
                if hasattr(eng, "run_with_progress"):
                    gen = eng.run_with_progress(
                        all_words, books=books_arg, dual_score=dual_score
                    )
                else:
                    _r = eng.run(all_words, books=books_arg, dual_score=dual_score)
                    gen = iter([(1.0, _r)])

                for pct, partial in gen:
                    progress_bar.progress(
                        float(pct),
                        text=f"Searching... {int(pct * 100)}%",
                    )
                    results = partial

                st.session_state["last_results"]          = results
                st.session_state["last_corpus"]           = eng.corpus_bytes()
                st.session_state["search_terms_snapshot"] = list(terms)
                progress_bar.progress(1.0, text="Search complete.")

            except Exception as exc:  # noqa: BLE001
                st.error(f"Search failed: {exc}")
                results = []

            if results:
                rows = []
                for m, cs in results:
                    rows.append({
                        "word":           m.word,
                        "verse":          m.verse,
                        "book":           m.book,
                        "skip":           m.skip,
                        "sequence":       m.sequence,
                        "hebert":         round(cs.hebert_score, 4),
                        "alephbert":      round(cs.alephbert_score, 4),
                        "consensus":      round(cs.consensus, 4),
                        "significant":    cs.is_significant,
                        "decadal_anchor": cs.is_decadal_anchor,
                    })
                df = pl.DataFrame(rows)
                st.success(f"{len(results)} significant hit(s) found.")

                anchor_hits = df.filter(pl.col("decadal_anchor"))
                if not anchor_hits.is_empty():
                    st.warning(
                        f"**Systemic Decadal Anchors detected** — "
                        f"{len(anchor_hits)} hit(s) co-located with "
                        f"תשפו / תשצ / תשצו."
                    )

                st.dataframe(df.to_pandas(), use_container_width=True)
                st.download_button(
                    "Export CSV",
                    df.write_csv(),
                    file_name="live_search.csv",
                    mime="text/csv",
                )
            else:
                st.info("No hits above threshold.")


# ══════════════════════════════════════════════════════════════════════════════
# ELS GRID
# ══════════════════════════════════════════════════════════════════════════════
elif panel == "ELS Grid":
    st.header("2D ELS Grid Viewer")

    results = st.session_state.get("last_results", [])
    corpus  = st.session_state.get("last_corpus", None)

    if not results or corpus is None:
        st.info("Run a Live Search first to populate the grid viewer.")
    else:
        # Build English lookup from the snapshot captured at search time
        _snapshot: list[dict] = st.session_state.get("search_terms_snapshot", [])
        _eng_map: dict[str, str] = {t["hebrew"]: t["english"] for t in _snapshot}

        match_labels = [
            f"{m.word}  |  {m.verse}  |  skip={m.skip}  |  consensus={cs.consensus:.3f}"
            for m, cs in results
        ]
        selected_label = st.selectbox("Select match", match_labels)
        idx = match_labels.index(selected_label)
        match, score = results[idx]

        english_label = _eng_map.get(match.word, "")

        # Prominent English header when a translation is available
        if english_label:
            st.subheader(f"{match.word} — {english_label}")
        else:
            st.subheader(match.word)

        context_chars = st.slider("Context window (chars)", 200, 2_000, 1_000, step=100)

        with st.spinner("Rendering grid …"):
            grid = _paint(corpus, match.start, match.skip, match.word,
                          english_label=english_label)

        st.markdown(
            f"**Grid width:** {grid.width}  |  "
            f"**Rows:** {grid.grid_array.shape[0]}  |  "
            f"**Primary cells:** {len(grid.word_cells)}  |  "
            f"**Cross-word hits:** {len(grid.crossword_hits)}"
        )

        if grid.crossword_hits:
            with st.expander("Cross-word detections"):
                for h in grid.crossword_hits[:20]:
                    st.code(f"{h.direction}  @ row={h.start_row} col={h.start_col}  "
                            f"→ '{h.letters}' ({h.length} letters)")

        st.components.v1.html(grid.html, height=min(800, grid.grid_array.shape[0] * 16 + 60), scrolling=True)


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK HUB
# ══════════════════════════════════════════════════════════════════════════════
elif panel == "Network Hub":
    st.header("Force-Directed Network Hub")
    st.caption(
        "Nodes = Verses (blue) · Search Tokens (orange) · Year Codes (green).  "
        "Edges = co-occurrence handshakes weighted by consensus score.  "
        "**Red node** = תשצ (2030 Wall) gravity well.  Red edges = Decadal Anchors."
    )

    results = st.session_state.get("last_results", [])
    if not results:
        st.info("Run a Live Search first to populate the network graph.")
    else:
        with st.spinner("Building graph …"):
            net = _get_network()
            net.ingest(results)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Nodes", net.node_count)
        with col_b:
            st.metric("Edges", net.edge_count)

        # Gravity well report for the 2030 Wall
        analyzer = _get_analyzer(net)
        gw = analyzer.gravity_well_report()
        if gw["degree"] > 0:
            dominant_label = "YES — dominant year node" if gw["is_dominant"] else "No"
            st.markdown(
                f"**2030 Wall (תשצ) Gravity Analysis** — "
                f"Degree: **{gw['degree']}** · "
                f"Weighted degree: **{gw['weighted_degree']:.3f}** · "
                f"Dominant: **{dominant_label}**"
            )
        else:
            st.info("תשצ (2030 Wall) node not yet present in the graph.  "
                    "Search for תשצ explicitly to anchor the gravity well.")

        # PyVis render
        with st.spinner("Rendering interactive graph …"):
            try:
                hub_path = _ROOT / "docs" / "network_hub.html"
                analyzer.render(output_path=hub_path)
                html_src = hub_path.read_text(encoding="utf-8")
                st.components.v1.html(html_src, height=720, scrolling=False)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Graph render failed: {exc}")

        # Top convergences table
        st.subheader("Top Verse Convergences")
        top_df = net.top_convergences(n=15)
        if not top_df.is_empty():
            st.dataframe(top_df.to_pandas(), use_container_width=True)

        # Betweenness centrality table
        with st.expander("Betweenness Centrality (all nodes)"):
            bc_df = net.betweenness_centrality()
            if not bc_df.is_empty():
                st.dataframe(bc_df.head(30).to_pandas(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHRONOS AUDIT
# ══════════════════════════════════════════════════════════════════════════════
elif panel == "Chronos Audit":
    st.header("Chronos Anchor Audit")
    st.caption(
        "Decadal reference frame: תשפו (5786 / 2026), "
        "תשצ (5790 / 2030), תשצו (5796 / 2036)."
    )

    docs_dir = _ROOT / "docs"
    report   = docs_dir / "decadal_horizon.md"

    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("No decadal_horizon.md found in docs/.  Run `bot/manager.py --commit` to generate.")

    results = st.session_state.get("last_results", [])
    if results:
        anchor_hits = [(m, cs) for m, cs in results if cs.is_decadal_anchor]
        if anchor_hits:
            st.subheader(f"Current session — {len(anchor_hits)} Decadal Anchor hit(s)")
            rows = [
                {
                    "word":     m.word,
                    "verse":    m.verse,
                    "skip":     m.skip,
                    "consensus": round(cs.consensus, 4),
                }
                for m, cs in anchor_hits
            ]
            st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True)
        else:
            st.info("No Systemic Decadal Anchors in the current session results.")
