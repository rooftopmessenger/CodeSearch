"""
archiver.py – Persistent Semantic Layer for ELS findings.

Stores high-significance ELS matches in a local ChromaDB vector database,
keyed by their HeBERT word embeddings, so that cross-run semantic clustering
and theme-based retrieval are possible without re-running the search engine.

Design notes
------------
Language separation is preserved in the vector store in two ways:

1. **Compact-encoding provenance** — Every archived match carries its `word`
   bytes in compact encoding (Hebrew bytes 1–27, English bytes 28–53).  The
   two ranges never overlap, so a Hebrew search term cannot be confused with
   an English one in the metadata filter layer.

2. **Embedding model gate** — HeBERT is a Hebrew-only model.  English KJV
   matches always carry ``hebert_score = 0.0`` (see ``engine._KJV_NT_BOOK_NAMES``
   guard).  ``archive_matches()`` rejects any match whose ``hebert_score == 0.0``,
   so the collection contains only Hebrew-validated entries.

Database layout
---------------
  data/chroma/          ← ChromaDB PersistentClient storage root
    chroma.sqlite3      ← HNSW index + metadata (auto-managed by ChromaDB)

Collection: ``bible_codes``
  - HNSW distance metric: cosine
  - Vector dimension: 768 (HeBERT hidden size)
  - Document field: ELS sequence (actual Hebrew letters from corpus)
  - Metadata fields: word, book, verse, skip, z_score, hebert_score,
                     semantic_z_score, is_significant, start, length
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb

import validator as _val
from engine import Match, _KJV_NT_BOOK_NAMES as _KJV_NT_BOOK_NAMES

# ── Constants ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_DB_DIR: Path = _PROJECT_ROOT / "data" / "chroma"
_COLLECTION_NAME: str = "bible_codes"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_client: chromadb.PersistentClient | None = None
_collection = None  # chromadb.Collection


def _get_collection():
    """Return the ChromaDB collection, creating it on first access."""
    global _client, _collection
    if _collection is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(_DB_DIR))
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _match_doc_id(m: Match) -> str:
    """
    Derive a stable 32-hex-character ID for a match.

    The ID is a SHA-256 digest of ``word|book|verse|skip|start`` so that
    the same physical ELS occurrence upserted from multiple runs does not
    create duplicate rows.
    """
    raw = f"{m.word}|{m.book}|{m.verse}|{m.skip}|{m.start}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ── Public API ────────────────────────────────────────────────────────────────

def archive_matches(matches: list[Match]) -> int:
    """
    Persist significant ELS matches to the ChromaDB vector store.

    **Filter rules** — a match is archived only when *all* of the following
    hold:

    * ``is_significant is True`` — corpus Z-score exceeded the significance
      threshold (default 3.0) set during the baseline run.
    * ``hebert_score > 0.0`` — HeBERT produced a real embedding for this hit.
      English KJV matches always have ``hebert_score = 0.0`` because HeBERT is
      a Hebrew-only model; they are excluded here to keep the vector space
      linguistically clean.

    The embedding stored for each match is the HeBERT unit-vector for the ELS
    *word* itself (retrieved from validator's in-process cache, which is populated
    during the original ``engine.search()`` run when ``validate=True``).  Storing
    the word embedding — rather than the context embedding — means that a query
    for the theme "Leadership" will surface matches whose ELS *words* are
    semantically related to leadership, regardless of the surrounding verse text.

    Parameters
    ----------
    matches :
        Full result list from ``ELSEngine.search()``, enriched with corpus
        Z-scores and ``is_significant`` flags via ``stats.apply_significance()``.

    Returns
    -------
    int
        Number of matches successfully written to the database (0 if none
        passed the filter).
    """
    significant = [
        m for m in matches
        if m.is_significant and m.hebert_score > 0.0
        and m.book not in _KJV_NT_BOOK_NAMES
    ]
    if not significant:
        return 0

    collection = _get_collection()

    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    documents: list[str] = []

    for m in significant:
        # The word embedding is already in validator's cache from the search run.
        # Calling _embed() a second time costs nothing (pure cache hit).
        emb_tensor = _val._embed(m.word)
        embedding: list[float] = emb_tensor.tolist()

        ids.append(_match_doc_id(m))
        embeddings.append(embedding)
        metadatas.append({
            "word": m.word,
            "book": m.book,
            "verse": m.verse,
            "skip": m.skip,
            "z_score": round(m.z_score, 6),
            "hebert_score": round(m.hebert_score, 6),
            "semantic_z_score": round(m.semantic_z_score, 6),
            "is_significant": int(m.is_significant),   # ChromaDB stores bool as int
            "start": m.start,
            "length": m.length,
        })
        documents.append(m.sequence)  # actual Hebrew letters from corpus

    # upsert: re-archiving the same match (same ID) does an in-place update
    # rather than creating a duplicate — safe to call on every run.
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    return len(ids)


def find_semantic_clusters(query_text: str, n_results: int = 5) -> list[dict]:
    """
    Return archived ELS matches semantically similar to *query_text*.

    Embeds *query_text* with HeBERT and queries the ``bible_codes`` collection
    for the nearest neighbours in cosine similarity space.  Works best with
    Hebrew thematic words (e.g. ``מנהיגות`` — leadership) but also accepts
    transliterated or translated English input; HeBERT will embed it as best
    it can.

    Parameters
    ----------
    query_text :
        A Hebrew or thematic text string to use as the search centre.
        Example: ``'מנהיג'`` (leader), ``'ברית שלם'`` (covenant of peace).
    n_results :
        Maximum number of results to return (default 5).  Automatically capped
        at the number of entries currently in the database.

    Returns
    -------
    list[dict]
        Each dict contains: ``word``, ``book``, ``verse``, ``skip``,
        ``z_score``, ``hebert_score``, ``semantic_z_score``,
        ``original_text`` (the ELS sequence), ``distance`` (cosine distance,
        lower = more similar).  Empty list if the database is empty.
    """
    collection = _get_collection()
    n_in_db = collection.count()
    if n_in_db == 0:
        return []

    actual_n = min(n_results, n_in_db)
    query_emb: list[float] = _val._embed(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=actual_n,
        include=["metadatas", "documents", "distances"],
    )

    out: list[dict] = []
    for meta, doc, dist in zip(
        results["metadatas"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        out.append({**meta, "original_text": doc, "distance": round(dist, 6)})
    return out


def db_stats() -> dict:
    """
    Return a summary of the current database state.

    Returns
    -------
    dict with keys:
        ``count``       – total number of archived matches
        ``collection``  – name of the ChromaDB collection
        ``db_dir``      – absolute path to the storage directory
    """
    collection = _get_collection()
    return {
        "count": collection.count(),
        "collection": _COLLECTION_NAME,
        "db_dir": str(_DB_DIR.resolve()),
    }
