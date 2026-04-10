"""n
stats.py – Statistical baseline for ELS significance testing via Monte Carlo.

Algorithm
---------
1. The real Torah compact bytes (1 byte per Hebrew letter) are shuffled using
   Fisher-Yates (via random.shuffle on a bytearray) to produce a sequence that
   preserves the exact character-frequency distribution of the original text but
   destroys all sequence information.

2. An ELS hit-count search is run on the shuffled bytes for each trial.
   This uses pure-Python bytes slicing + StringZilla SIMD find() — no PyTorch
   tensor construction per trial — so it is fast enough for 100 trials.

3. After n_trials, the hit-count distribution gives us:
     μ (hit_count_mean)  — expected random hits at these skip parameters
     σ (hit_count_std)   — spread of random hits

4. For the real Torah search result:
     Z_corpus = (real_hit_count − μ) / σ

   This corpus-level Z-score is assigned to every individual Match via
   ``apply_significance()``.

5. Per-Match Semantic Significance (score_sample_size > 0)
   --------------------------------------------------------
   When ``score_sample_size`` is passed to ``run_monte_carlo``, up to that many
   random hit (word, fake-context) pairs are extracted across all shuffle trials
   and batch-scored through HeBERT.  This builds a null distribution of HeBERT
   scores for random ELS hits:

     μ_score (BaselineResult.score_mean)  — expected HeBERT score in shuffled text
     σ_score (BaselineResult.score_std)   — spread of random HeBERT scores

   ``apply_significance()`` then assigns a per-match Z to each real Match:
     Z_semantic = (match.hebert_score − μ_score) / σ_score

   This captures whether a match's semantic coherence is unusually high compared
   to random ELS hits — not just whether the total hit count is unusual.
"""

from __future__ import annotations

import random
import math
import statistics
from dataclasses import dataclass, replace as dc_replace
from typing import TYPE_CHECKING, Sequence

from stringzilla import Str as SZStr
from tqdm import tqdm

if TYPE_CHECKING:
    from engine import Match

# Compact encoding base — mirrors engine._HEB_BASE
_HEB_BASE = 0x05D0
# English compact offset — mirrors engine._ENG_COMPACT_OFFSET (A=28, B=29, … Z=53)
# Hebrew non-final consonants use bytes 1–27 (alef=1 … tav=27).
# English letters use bytes 28–53 to ensure no overlap.
_ENG_COMPACT_OFFSET = 27


def _decode_compact_byte(b: int) -> str:
    """Decode one compact byte: Hebrew (1–22) → Unicode consonant; English (23–48) → ‘A’–‘Z’."""
    if b > _ENG_COMPACT_OFFSET:
        return chr(b - _ENG_COMPACT_OFFSET - 1 + 65)
    return chr(b + _HEB_BASE - 1)


# Avoid circular import: _compact is imported lazily inside run_monte_carlo()


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BaselineResult:
    """Statistics collected from Monte Carlo shuffled-text runs."""
    n_trials: int
    hit_count_mean: float       # μ of hit counts across all trials
    hit_count_std: float        # σ of hit counts across all trials
    score_mean: float = 0.0     # μ of HeBERT scores (0.0 if not tracked)
    score_std: float = 0.0      # σ of HeBERT scores (0.0 if not tracked)


# ── Shuffle ────────────────────────────────────────────────────────────────────

def shuffle_bytes(data: bytes, rng: random.Random) -> bytes:
    """
    Return a Fisher-Yates shuffled copy of *data*.

    Uses Python's built-in ``random.shuffle`` on a ``bytearray``,
    which is O(n) and implemented in C.  The character-frequency
    distribution is preserved exactly.
    """
    arr = bytearray(data)
    rng.shuffle(arr)
    return bytes(arr)


# ── Lightweight hit counter (no provenance, no tensor) ────────────────────────

def _count_hits(
    compact_bytes: bytes,
    compact_words: list[bytes],
    min_skip: int,
    max_skip: int,
    long_skip: bool = False,
) -> int:
    """
    Count total ELS hits in *compact_bytes* using StringZilla only.

    No PyTorch tensors, no index map — just SIMD byte searching.
    Both forward and backward skip directions are searched.

    When *long_skip* is True, skip values with abs(skip) < 10 are excluded so
    that the Monte Carlo null distribution matches the long-skip-only real search.
    """
    # Reverse once; all backward-skip strides reuse this.
    compact_rev = compact_bytes[::-1]
    count = 0

    for skip in range(min_skip, max_skip + 1):
        if long_skip and skip < 10:
            continue
        # Forward: positions 0, skip, 2*skip, …
        fwd_bytes = compact_bytes[::skip]
        sz_fwd = SZStr(fwd_bytes)

        # Backward: positions n-1, n-1-skip, n-1-2*skip, …
        bwd_bytes = compact_rev[::skip]
        sz_bwd = SZStr(bwd_bytes)

        for cword in compact_words:
            # Forward scan
            pos = 0
            while True:
                idx = sz_fwd.find(cword, pos)
                if idx < 0:
                    break
                count += 1
                pos = idx + 1

            # Backward scan
            pos = 0
            while True:
                idx = sz_bwd.find(cword, pos)
                if idx < 0:
                    break
                count += 1
                pos = idx + 1

    return count


# ── Context extraction for shuffle hits ───────────────────────────────────────

def _collect_hit_pairs(
    compact_bytes: bytes,
    compact_words: list[bytes],
    min_skip: int,
    max_skip: int,
    max_pairs: int,
    context_window: int = 100,
) -> list[tuple[str, str]]:
    """
    Find up to *max_pairs* ELS hits in *compact_bytes* and return them as
    ``(word_str, context_str)`` pairs ready for HeBERT scoring.

    ``context_str`` is a window of *context_window* letters on each side of the
    real start position — decoded from the (shuffled) compact bytes to Hebrew
    Unicode.  Because the bytes come from a frequency-preserving shuffle, the
    context looks like scrambled Hebrew, which is exactly the null distribution
    we want for semantic scoring.

    Collection stops as soon as *max_pairs* pairs are accumulated so that we
    don't over-sample from early skip values.
    """
    pairs: list[tuple[str, str]] = []
    n = len(compact_bytes)
    compact_rev = compact_bytes[::-1]

    for skip in range(min_skip, max_skip + 1):
        if len(pairs) >= max_pairs:
            break

        fwd_bytes = compact_bytes[::skip]
        sz_fwd    = SZStr(fwd_bytes)
        bwd_bytes = compact_rev[::skip]
        sz_bwd    = SZStr(bwd_bytes)

        for cword in compact_words:
            word_str = "".join(_decode_compact_byte(b) for b in cword)

            # ── Forward hits ──────────────────────────────────────────────────
            pos = 0
            while len(pairs) < max_pairs:
                idx = sz_fwd.find(cword, pos)
                if idx < 0:
                    break
                real_start = idx * skip
                lo = max(0, real_start - context_window)
                hi = min(n, real_start + len(cword) * skip + context_window)
                ctx = "".join(
                    _decode_compact_byte(b)
                    for b in compact_bytes[lo:hi]
                    if 1 <= b <= 53
                )
                if ctx:
                    pairs.append((word_str, ctx))
                pos = idx + 1

            # ── Backward hits ─────────────────────────────────────────────────
            pos = 0
            while len(pairs) < max_pairs:
                idx = sz_bwd.find(cword, pos)
                if idx < 0:
                    break
                real_start = n - 1 - idx * skip
                lo = max(0, real_start - context_window)
                hi = min(n, real_start + context_window)
                ctx = "".join(
                    _decode_compact_byte(b)
                    for b in compact_bytes[lo:hi]
                    if 1 <= b <= 53
                )
                if ctx:
                    pairs.append((word_str, ctx))
                pos = idx + 1

    return pairs


# ── Monte Carlo runner ─────────────────────────────────────────────────────────

def run_monte_carlo(
    compact_bytes: bytes,
    words: Sequence[str],
    min_skip: int,
    max_skip: int,
    n_trials: int = 100,
    *,
    seed: int = 42,
    show_progress: bool = True,
    score_sample_size: int = 0,
    score_batch_size: int = 32,
    long_skip: bool = False,
) -> BaselineResult:
    """
    Shuffle the corpus *n_trials* times and count ELS hits on each shuffle.

    Parameters
    ----------
    compact_bytes :
        The 1-byte-per-letter compact corpus from ``ELSEngine.corpus_bytes_for()``.
    words :
        Hebrew Unicode search terms (same list used for the real search).
    min_skip, max_skip :
        Skip range — must match the real search parameters.
    n_trials :
        Number of random shuffles to run (default 100).
    seed :
        RNG seed for reproducibility (default 42).
    show_progress :
        Show a tqdm progress bar over trials.
    score_sample_size : int
        If > 0, collect up to this many random hit (word, fake-context) pairs
        across all trials and batch-score them with HeBERT to build a semantic
        null distribution.  A value of 500 is a good default — large enough for
        stable μ/σ estimates without excessive HeBERT calls.  Set to 0 (default)
        to skip semantic scoring entirely.
    score_batch_size : int
        Mini-batch size used when calling HeBERT on the collected pairs (default 32).

    Returns
    -------
    BaselineResult
        Contains hit_count_mean, hit_count_std.  When *score_sample_size* > 0,
        score_mean and score_std are also populated from the HeBERT null distribution.
    """
    # Lazy import to avoid circular dependency at module load time
    from engine import _compact as _enc

    compact_words = [_enc(w) for w in words]
    rng = random.Random(seed)
    hit_counts: list[int] = []
    sampled_pairs: list[tuple[str, str]] = []

    for _ in tqdm(range(n_trials), desc="Monte Carlo baseline", disable=not show_progress):
        shuffled = shuffle_bytes(compact_bytes, rng)
        n = _count_hits(shuffled, compact_words, min_skip, max_skip, long_skip=long_skip)
        hit_counts.append(n)

        # Collect semantic scoring samples across trials until cap is reached
        if score_sample_size > 0 and len(sampled_pairs) < score_sample_size:
            remaining = score_sample_size - len(sampled_pairs)
            new_pairs = _collect_hit_pairs(
                shuffled, compact_words, min_skip, max_skip, remaining
            )
            sampled_pairs.extend(new_pairs)

    mean = statistics.mean(hit_counts)
    std  = statistics.stdev(hit_counts) if len(hit_counts) > 1 else 0.0

    score_mean = 0.0
    score_std  = 0.0
    if sampled_pairs:
        import validator as _val
        print(
            f"Scoring {len(sampled_pairs)} random hit samples with HeBERT "
            f"(batch_size={score_batch_size}) …"
        )
        rand_scores = _val.score_pairs_batch(sampled_pairs, batch_size=score_batch_size)
        score_mean = statistics.mean(rand_scores)
        score_std  = (
            statistics.stdev(rand_scores) if len(rand_scores) > 1 else 0.0
        )

    return BaselineResult(
        n_trials=n_trials,
        hit_count_mean=mean,
        hit_count_std=std,
        score_mean=score_mean,
        score_std=score_std,
    )


# ── Z-score helpers ────────────────────────────────────────────────────────────

def compute_z_score(value: float, mean: float, std: float) -> float:
    """Z = (value − mean) / std.

    Returns 0.0 when std is effectively zero (< 1e-6) to avoid ZeroDivisionError
    and prevent astronomical Z values caused by floating-point residuals.
    """
    if std < 1e-6:
        return 0.0
    return (value - mean) / std


# ── Significance enrichment ────────────────────────────────────────────────────

def apply_significance(
    matches: list[Match],
    baseline: BaselineResult,
    real_hit_count: int,
    *,
    z_threshold: float = 3.0,
) -> list[Match]:
    """
    Return a new list of Match objects enriched with significance fields.

    **Corpus-level Z** (same for all matches in this search):
        Z_corpus = (real_hit_count − μ_hits) / σ_hits

        Sets ``z_score`` and ``is_significant`` on every match.

    **Per-match semantic Z** (only when HeBERT scoring was run on both the real
    search and the Monte Carlo baseline, i.e. ``baseline.score_std > 0``):
        Z_semantic = (match.hebert_score − baseline.score_mean) / baseline.score_std

        Sets ``semantic_z_score`` on each match individually.

    Parameters
    ----------
    matches :
        The real Match objects from ``ELSEngine.search()``.
    baseline :
        The ``BaselineResult`` from ``run_monte_carlo()``.
    real_hit_count :
        ``len(matches)`` — total hits in the real corpus.
    z_threshold :
        Significance cutoff used for ``is_significant`` (default 3.0).

    Returns
    -------
    list[Match]
        New frozen-dataclass instances with z_score, is_significant, and
        (when available) semantic_z_score set.
    """
    z_corpus   = compute_z_score(real_hit_count, baseline.hit_count_mean, baseline.hit_count_std)
    significant = z_corpus > z_threshold
    has_sem     = baseline.score_std > 0.0

    enriched = []
    for m in matches:
        sem_z = (
            compute_z_score(m.hebert_score, baseline.score_mean, baseline.score_std)
            if has_sem
            else 0.0
        )
        enriched.append(
            dc_replace(m, z_score=z_corpus, is_significant=significant, semantic_z_score=sem_z)
        )
    return enriched
