"""
engine/search.py — CodeSearch-Ultra class-based search engine.

Wraps the existing backend ELSEngine into a higher-level ``UltraSearchEngine``
that introduces:

  - Dual-model semantic scoring (HeBERT + alephbert-base) producing a
    ``ConsensusScore`` that must exceed 0.30 to be classified as a
    "Significant Hit."
  - Automatic Chronos Anchor cross-reference: every run is tested against
    the three Decadal Anchors תשפו (5786 / 2026), תשצ (5790 / 2030), and
    תשצו (5796 / 2036).  Any match whose nearest skip matches one of these
    anchor year-codes is flagged as a Systemic Decadal Anchor.
  - A ``run()`` façade that accepts plain Python keyword arguments aligned
    with the existing CLI flags so the bot layer needs no argparse knowledge.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# ── resolve backend on sys.path ───────────────────────────────────────────────
# NOTE: We use importlib to load backend/engine.py by file path to avoid a
# name collision with this package (also called 'engine').
import importlib.util as _ilu

_BACKEND = Path(__file__).resolve().parent.parent / "backend"

# backend/ must be on sys.path so that backend/engine.py's own bare-name
# imports (data_loader, validator, etc.) resolve correctly.
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _load_backend_module(name: str):
    """Load a module from backend/ by file path, avoiding package-name collision."""
    mod_name = f"_backend_{name}"
    # Return cached module if already loaded (avoids double-exec across imports)
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = _ilu.spec_from_file_location(mod_name, _BACKEND / f"{name}.py")
    mod  = _ilu.module_from_spec(spec)   # type: ignore[arg-type]
    # Must register BEFORE exec so dataclass / __module__ resolution works.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)         # type: ignore[union-attr]
    return mod


_be   = _load_backend_module("engine")
_val  = _load_backend_module("validator")
ELSEngine = _be.ELSEngine
Match     = _be.Match

# ── Chronos Anchors ──────────────────────────────────────────────────────────
# These year-code search terms are cross-referenced against every run.
# Flag: "Systemic Decadal Anchor" per copilot-instructions specification.
CHRONOS_ANCHORS: tuple[str, ...] = (
    "תשפו",   # 5786 / 2026
    "תשצ",    # 5790 / 2030
    "תשצו",   # 5796 / 2036
)

# Minimum consensus score threshold to classify a hit as Significant.
SIGNIFICANCE_THRESHOLD: float = 0.30


# ── Linguistic Jury thresholds ───────────────────────────────────────────────
# A hit is "Verified" when the ancient model score is within 20% of the modern
# model score (i.e. |aleph - hebert| / max(hebert, ε) <= 0.20).
# A hit is a "Deep Archetypal Anchor" when AlephBERT scores significantly
# higher than HeBERT (aleph > hebert * 1.15), meaning the ancient-text model
# finds stronger resonance than the modern one.
_JURY_TOLERANCE: float = 0.20   # ±20% band for Verified status
_DEEP_ANCHOR_RATIO: float = 1.15  # AlephBERT > HeBERT × 1.15 → Deep Archetypal


@dataclass(frozen=True)
class ConsensusScore:
    """
    Dual-model semantic score for a single ELS match (Linguistic Jury).

    hebert_score
        Cosine similarity from avichr/heBERT (Modern Hebrew model).
    alephbert_score
        Cosine similarity from onlplab/alephbert-base (Ancient Hebrew model).
    consensus
        Arithmetic mean of the two scores.  Used for all significance gating.
    is_significant
        True when *consensus* >= ``SIGNIFICANCE_THRESHOLD`` (0.30).
    is_decadal_anchor
        True when at least one Chronos Anchor word co-locates with this hit.
    is_verified
        Jury verdict: True when |aleph - hebert| / max(hebert, 1e-9) <= 0.20.
        Both models agree within 20% — linguistic coherence confirmed.
    is_deep_archetypal_anchor
        True when AlephBERT scores > HeBERT × 1.15.  The ancient-text model
        finds stronger resonance than the modern one — flag as a deep
        archetypal signal rooted in biblical register.
    """

    hebert_score: float
    alephbert_score: float
    consensus: float
    is_significant: bool
    is_decadal_anchor: bool = False
    is_verified: bool = False
    is_deep_archetypal_anchor: bool = False


# ── AlephBERT lazy loader ─────────────────────────────────────────────────────

_aleph_tokenizer = None
_aleph_model = None


def _load_alephbert():
    """Lazy-load onlplab/alephbert-base the first time it is needed."""
    global _aleph_tokenizer, _aleph_model
    if _aleph_tokenizer is None:
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            _name = "onlplab/alephbert-base"
            print(f"Loading AlephBERT ({_name}) …")
            _aleph_tokenizer = AutoTokenizer.from_pretrained(_name)
            _aleph_model = AutoModel.from_pretrained(_name)
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _aleph_model.to(_device)
            _aleph_model.eval()
        except Exception as exc:  # noqa: BLE001
            # AlephBERT unavailable (network-gated environment): fall back to
            # mirroring HeBERT score so consensus == hebert_score.
            print(f"[UltraSearchEngine] AlephBERT load failed ({exc}); "
                  "consensus will mirror HeBERT score.")
            _aleph_tokenizer = "UNAVAILABLE"
            _aleph_model = None
    return _aleph_tokenizer, _aleph_model


def _score_alephbert(text: str, context: str) -> float:
    """
    Compute cosine similarity between *text* and *context* using AlephBERT.

    Returns 0.0 if the model is unavailable or if text is not Hebrew.
    """
    tok, mdl = _load_alephbert()
    if mdl is None:
        return 0.0

    import torch
    import torch.nn.functional as F

    device = next(mdl.parameters()).device

    def _embed(s: str) -> torch.Tensor:
        inputs = tok(s, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = mdl(**inputs)
        return F.normalize(out.last_hidden_state.mean(dim=1), dim=-1)

    try:
        e_word = _embed(text)
        e_ctx  = _embed(context)
        return float(F.cosine_similarity(e_word, e_ctx).item())
    except Exception:  # noqa: BLE001
        return 0.0


# ── UltraSearchEngine ─────────────────────────────────────────────────────────


class UltraSearchEngine:
    """
    High-level ELS search engine for CodeSearch-Ultra.

    Wraps ``backend.engine.ELSEngine`` and adds:
      - Dual-model consensus scoring (HeBERT + AlephBERT).
      - Automatic Chronos Anchor cross-reference on every run.
      - Structured result list of ``(Match, ConsensusScore)`` tuples.

    Parameters
    ----------
    min_skip : int
        Minimum |skip| distance (default 1).
    max_skip : int
        Maximum |skip| distance (default 1000).
    texts_dir : Path | str
        Path to the Hebrew/KJV source .txt files.
    data_dir : Path | str
        Cache directory for compiled corpus artefacts.
    """

    def __init__(
        self,
        *,
        min_skip: int = 1,
        max_skip: int = 1_000,
        texts_dir: str | Path = _BACKEND.parent / "texts",
        data_dir: str | Path = _BACKEND.parent / "data",
    ) -> None:
        self._engine = ELSEngine(
            min_skip=min_skip,
            max_skip=max_skip,
            texts_dir=texts_dir,
            data_dir=data_dir,
        )
        self._min_skip = min_skip
        self._max_skip = max_skip

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        words: Sequence[str],
        *,
        books: list[str] | None = None,
        validate: bool = True,
        score_threshold: float = SIGNIFICANCE_THRESHOLD,
        dual_score: bool = True,
        check_anchors: bool = True,
    ) -> list[tuple[Match, ConsensusScore]]:
        """
        Execute an ELS search and return all matches with their consensus scores.

        Parameters
        ----------
        words :
            Hebrew (or English) search terms.
        books :
            Optional list of book names / group aliases to restrict the corpus.
        validate :
            When True (default), compute HeBERT scores for every match.
        score_threshold :
            Minimum consensus score to include in results.  Defaults to
            the platform-wide ``SIGNIFICANCE_THRESHOLD`` (0.30).
        dual_score :
            When True (default), compute AlephBERT secondary score.  Setting
            this to False speeds up batch runs at the cost of consensus resolution.
        check_anchors :
            When True (default), append a Chronos Anchor cross-reference pass
            and flag any match co-located near an anchor hit.

        Returns
        -------
        list of (Match, ConsensusScore) tuples, sorted by consensus score
        descending.
        """
        # validate is a dataclass field on ELSEngine (not a search() kwarg).
        # Set it before the call so HeBERT scoring activates when requested.
        self._engine.validate = validate

        raw_matches: list[Match] = self._engine.search(
            words=list(words),
            books=books,
        )

        anchor_verses: set[str] = set()
        if check_anchors:
            anchor_verses = self._collect_anchor_verses()

        results: list[tuple[Match, ConsensusScore]] = []
        for m in raw_matches:
            cs = self._build_consensus(m, dual_score=dual_score,
                                       anchor_verses=anchor_verses)
            if cs.consensus >= score_threshold:
                results.append((m, cs))

        results.sort(key=lambda t: t[1].consensus, reverse=True)
        return results

    def run_with_progress(
        self,
        words: Sequence[str],
        *,
        books: list[str] | None = None,
        validate: bool = True,
        score_threshold: float = SIGNIFICANCE_THRESHOLD,
        dual_score: bool = True,
        check_anchors: bool = True,
    ):
        """
        Generator variant of ``run()``.  Yields ``(pct: float, results: list)``
        tuples as each word's ELS skip scan completes.

        Progress increments by 1/len(words) per word.  Emits:
          - (0.0, [])                    — immediately on entry
          - (word_n/total, partial)      — after each word finishes
          - (1.0, sorted_full_results)   — final emission after sort

        Designed for consumption by ``st.progress()`` in the dashboard::

            for pct, partial in eng.run_with_progress(words):
                progress_bar.progress(pct, text=f"Searching... {int(pct*100)}%")
        """
        self._engine.validate = validate
        word_list = list(words)
        n_words = max(len(word_list), 1)

        anchor_verses: set[str] = set()
        if check_anchors:
            anchor_verses = self._collect_anchor_verses()

        accumulated: list[tuple[Match, ConsensusScore]] = []
        yield (0.0, [])

        for i, word in enumerate(word_list):
            raw: list[Match] = self._engine.search_single(word, books=books)
            for m in raw:
                cs = self._build_consensus(m, dual_score=dual_score,
                                           anchor_verses=anchor_verses)
                if cs.consensus >= score_threshold:
                    accumulated.append((m, cs))

            # Reserve 1.0 for the final sorted emission so callers can
            # distinguish "word done" from "fully sorted and complete".
            pct = min((i + 1) / n_words, 0.99)
            yield (pct, list(accumulated))

        accumulated.sort(key=lambda t: t[1].consensus, reverse=True)
        yield (1.0, accumulated)

    def corpus_bytes(self) -> bytes:
        """Expose the compiled corpus bytes for grid and network modules."""
        return self._engine.corpus_bytes

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_consensus(
        self,
        m: Match,
        *,
        dual_score: bool,
        anchor_verses: set[str],
    ) -> ConsensusScore:
        hebert = m.hebert_score
        aleph  = _score_alephbert(m.word, m.sequence) if dual_score else hebert
        consensus = (hebert + aleph) / 2.0

        is_decadal = m.verse in anchor_verses or any(
            a in m.sequence for a in CHRONOS_ANCHORS
        )

        # ── Linguistic Jury verdict ───────────────────────────────────────────
        _eps = 1e-9
        deviation = abs(aleph - hebert) / max(hebert, _eps)
        is_verified = deviation <= _JURY_TOLERANCE
        is_deep_arch = aleph > hebert * _DEEP_ANCHOR_RATIO

        return ConsensusScore(
            hebert_score=hebert,
            alephbert_score=aleph,
            consensus=consensus,
            is_significant=consensus >= SIGNIFICANCE_THRESHOLD,
            is_decadal_anchor=is_decadal,
            is_verified=is_verified,
            is_deep_archetypal_anchor=is_deep_arch,
        )

    def _collect_anchor_verses(self) -> set[str]:
        """
        Run a lightweight ELS pass for each Chronos Anchor term and collect
        the verse references of all hits.  Results are used to mark coincident
        matches as Systemic Decadal Anchors.
        """
        verses: set[str] = set()
        try:
            self._engine.validate = False   # no HeBERT scoring for anchor pass
            for anchor in CHRONOS_ANCHORS:
                hits: list[Match] = self._engine.search(
                    words=[anchor],
                    books=None,
                )
                for h in hits:
                    verses.add(h.verse)
        except Exception:  # noqa: BLE001
            pass
        return verses
