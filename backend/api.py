"""
api.py – FastAPI bridge between the CodeSearch engine and the frontend.

Endpoints
---------
POST /search
    Run an ELS search and return matches as JSON.

GET /discover
    Query the ChromaDB semantic archive for thematically related ELS findings.

Static serving
--------------
The ``frontend/`` folder is mounted at ``/`` so the React / HTML UI is served
directly from this process.  If ``frontend/`` does not exist the static mount
is skipped (useful in headless/API-only deployments).

Usage
-----
From the project root::

    uvicorn backend.api:app --reload

Or from the ``backend/`` directory::

    uvicorn api:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── Ensure backend modules are importable regardless of launch CWD ─────────────
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine as _engine
import archiver as _archiver

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CodeSearch API",
    version="0.1.0",
    description=(
        "High-speed Bible Code (ELS) search engine with semantic clustering. "
        "Backed by StringZilla SIMD search, HeBERT / English-BERT validation, "
        "and ChromaDB semantic archive."
    ),
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# ── Lazy corpus cache ─────────────────────────────────────────────────────────
# ELSEngine loads and parses the full corpus on construction (~0.5 s).
# Cache a single instance so repeated /search calls don't re-parse on every
# request.  NOTE: this is a simple in-process singleton; for multi-worker
# deployments use a proper startup event or shared memory.
_cached_engine: _engine.ELSEngine | None = None


def _get_engine(
    min_skip: int,
    max_skip: int,
    validate: bool,
) -> _engine.ELSEngine:
    """Return a cached ELSEngine, rebuilding only if key parameters changed."""
    global _cached_engine
    if (
        _cached_engine is None
        or _cached_engine.min_skip != min_skip
        or _cached_engine.max_skip != max_skip
        or _cached_engine.validate != validate
    ):
        _cached_engine = _engine.ELSEngine(
            min_skip=min_skip,
            max_skip=max_skip,
            validate=validate,
            show_load_progress=False,
        )
    return _cached_engine


# ── Request / Response schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    words: list[str] = Field(..., min_length=1, description="ELS search terms.")
    books: list[str] | None = Field(None, description="Book filter (None = all books).")
    min_skip: int = Field(1, ge=1, description="Minimum ELS skip distance.")
    max_skip: int = Field(1000, ge=1, description="Maximum ELS skip distance.")
    run_validate: bool = Field(False, description="Score matches with HeBERT / English BERT.")
    top: int = Field(0, ge=0, description="Return only top-N matches sorted by |skip| (0 = all).")


class MatchOut(BaseModel):
    word: str
    book: str
    verse: str
    skip: int
    start: int
    sequence: str
    hebert_score: float
    z_score: float
    is_significant: bool


class SearchResponse(BaseModel):
    count: int
    wall_secs: float
    matches: list[MatchOut]


# ── /search ───────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, summary="Run ELS search")
def search(req: SearchRequest) -> SearchResponse:
    """
    Search for equidistant letter sequences across the specified corpus.

    Returns all matches sorted by absolute skip value, optionally capped at
    *top* results.
    """
    try:
        eng = _get_engine(req.min_skip, req.max_skip, req.run_validate)
        matches = eng.search(
            req.words,
            books=req.books or None,
            show_progress=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if req.top > 0:
        matches = sorted(matches, key=lambda m: abs(m.skip))[: req.top]

    return SearchResponse(
        count=len(matches),
        wall_secs=eng.last_wall_secs,
        matches=[
            MatchOut(
                word=m.word,
                book=m.book,
                verse=m.verse,
                skip=m.skip,
                start=m.start,
                sequence=m.sequence,
                hebert_score=m.hebert_score,
                z_score=m.z_score,
                is_significant=m.is_significant,
            )
            for m in matches
        ],
    )


# ── /discover ─────────────────────────────────────────────────────────────────

@app.get("/discover", summary="Semantic cluster lookup")
def discover(
    query: str = Query(..., description="Hebrew or English theme text to search for."),
    n_results: int = Query(5, ge=1, le=50, description="Number of nearest neighbours to return."),
) -> dict[str, Any]:
    """
    Query the ChromaDB semantic archive for ELS matches thematically similar to
    *query*.  Embeds *query* with the appropriate BERT model (HeBERT for Hebrew,
    all-mpnet-base-v2 for English) and returns the nearest neighbours by cosine
    similarity.
    """
    try:
        results = _archiver.find_semantic_clusters(query, n_results=n_results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"count": len(results), "results": results}


# ── /db-stats ─────────────────────────────────────────────────────────────────

@app.get("/db-stats", summary="ChromaDB collection statistics")
def db_stats() -> dict[str, Any]:
    """Return the current ChromaDB collection size and storage path."""
    try:
        return _archiver.db_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Static frontend (mount last so API routes take precedence) ────────────────
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
