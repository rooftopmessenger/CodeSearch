"""
validator.py – HeBERT-powered semantic scoring for ELS matches.

For each ELS match, ``score_match`` computes the cosine similarity between:
  - the embedding of the ELS word itself (what was found)
  - the mean embedding of the surrounding context text (3 verses)

A high score ( → 1.0) means the ELS word is semantically coherent with its
immediate textual neighbourhood.  A low score ( → 0.0) means the hit is
likely a random statistical artefact.

Model
-----
  avichr/heBERT  (Hebrew BERT, uncased)
  https://huggingface.co/avichr/heBERT

Embeddings are taken from the mean of the last hidden state (token embeddings),
which is a standard sentence-level representation for BERT-family models.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ── Model singletons ─────────────────────────────────────────────────────────

# Hebrew model (HeBERT)
_MODEL_NAME = "avichr/heBERT"
_tokenizer: AutoTokenizer | None = None
_model: AutoModel | None = None

# English model (sentence-transformers/all-mpnet-base-v2, 768-dim — cached locally)
_ENG_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_eng_tokenizer: AutoTokenizer | None = None
_eng_model: AutoModel | None = None

# Shared device — computed once at module load so both models use the same backend.
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Embedding caches ───────────────────────────────────────────────────────────
# Separate caches per model so Hebrew and English embeddings never collide.
_embed_cache: dict[str, torch.Tensor] = {}     # HeBERT embeddings
_cache_hits: int = 0
_cache_misses: int = 0
_eng_embed_cache: dict[str, torch.Tensor] = {} # English BERT embeddings
_eng_cache_hits: int = 0
_eng_cache_misses: int = 0


def clear_embed_cache() -> None:
    """Discard all cached embeddings (both models) and reset hit/miss counters."""
    global _cache_hits, _cache_misses, _eng_cache_hits, _eng_cache_misses
    _embed_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    _eng_embed_cache.clear()
    _eng_cache_hits = 0
    _eng_cache_misses = 0


def embed_cache_stats() -> dict[str, int]:
    """
    Return a snapshot of embedding cache usage for both models.

    Returns
    -------
    dict with keys:
        ``size``        – HeBERT cache entries
        ``hits``        – HeBERT cache hits
        ``misses``      – HeBERT cache misses
        ``eng_size``    – English BERT cache entries
        ``eng_hits``    – English BERT cache hits
        ``eng_misses``  – English BERT cache misses
    """
    return {
        "size": len(_embed_cache),
        "hits": _cache_hits,
        "misses": _cache_misses,
        "eng_size": len(_eng_embed_cache),
        "eng_hits": _eng_cache_hits,
        "eng_misses": _eng_cache_misses,
    }

def _load_model() -> tuple[AutoTokenizer, AutoModel, torch.device]:
    """Lazy-load HeBERT once and cache the result."""
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"Loading HeBERT ({_MODEL_NAME}) …")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME)
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device


def _load_eng_model() -> tuple[AutoTokenizer, AutoModel, torch.device]:
    """Lazy-load the English BERT model once."""
    global _eng_tokenizer, _eng_model
    if _eng_tokenizer is None:
        print(f"Loading English BERT ({_ENG_MODEL_NAME}) …")
        _eng_tokenizer = AutoTokenizer.from_pretrained(_ENG_MODEL_NAME)
        _eng_model = AutoModel.from_pretrained(_ENG_MODEL_NAME)
        _eng_model.to(_device)
        _eng_model.eval()
    return _eng_tokenizer, _eng_model, _device


def _is_english_text(text: str) -> bool:
    """
    Return True if *text* consists primarily of ASCII alphabetic characters.

    ELS search words are always purely one language (Hebrew bytes 1–27 or
    English bytes 28–53 in compact encoding), so a simple ASCII ratio check is
    sufficient to route to the correct embedding model.
    """
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    return sum(1 for c in alpha if ord(c) < 128) / len(alpha) > 0.8


# ── Core embedding helpers ────────────────────────────────────────────────────

def _embed(text: str) -> torch.Tensor:
    """
    Return a normalised 1-D sentence embedding for *text* using HeBERT.

    The embedding is the mean of all non-[PAD] token hidden states from the
    last encoder layer, then L2-normalised to unit length.

    Parameters
    ----------
    text : str
        Hebrew Unicode text to embed (verse text, ELS word, etc.)

    Returns
    -------
    torch.Tensor  shape (hidden_dim,), dtype float32, L2-norm == 1.0
    """
    tokenizer, model, device = _load_model()

    global _cache_hits, _cache_misses
    if text in _embed_cache:
        _cache_hits += 1
        return _embed_cache[text]
    _cache_misses += 1

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state: (batch=1, seq_len, hidden_dim)
    hidden = outputs.last_hidden_state[0]          # (seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"][0]   # (seq_len,)

    # Mean-pool over non-padding tokens
    mask = attention_mask.unsqueeze(-1).float()    # (seq_len, 1)
    summed = (hidden * mask).sum(dim=0)            # (hidden_dim,)
    mean_emb = summed / mask.sum()                 # (hidden_dim,)

    result = F.normalize(mean_emb, dim=0)           # unit vector
    _embed_cache[text] = result.cpu()               # store on CPU to conserve GPU memory
    return result


def _embed_english(text: str) -> torch.Tensor:
    """
    Return a normalised 1-D sentence embedding for *text* using the English BERT model.

    Mirrors the mean-pool + L2-normalise strategy used by :func:`_embed` so
    that cosine similarity scores are computed in the same way for both languages.
    Uses a separate cache (``_eng_embed_cache``) from the Hebrew model.
    """
    tokenizer, model, device = _load_eng_model()

    global _eng_cache_hits, _eng_cache_misses
    if text in _eng_embed_cache:
        _eng_cache_hits += 1
        return _eng_embed_cache[text]
    _eng_cache_misses += 1

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]           # (seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"][0]    # (seq_len,)
    mask = attention_mask.unsqueeze(-1).float()     # (seq_len, 1)
    summed = (hidden * mask).sum(dim=0)             # (hidden_dim,)
    mean_emb = summed / mask.sum()                  # (hidden_dim,)

    result = F.normalize(mean_emb, dim=0)           # unit vector
    _eng_embed_cache[text] = result.cpu()
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def score_match(els_word: str, context_text: str) -> float:
    """
    Compute the cosine similarity between *els_word* and *context_text*.

    Both inputs are embedded with HeBERT.  Because the embeddings are
    L2-normalised, cosine similarity reduces to a dot product and is
    guaranteed to be in [−1, 1].  In practice it stays in [0, 1] for
    Hebrew text.

    Parameters
    ----------
    els_word : str
        The Hebrew word found by the ELS search (Unicode consonants).
    context_text : str
        The concatenated text of the surrounding verses (≈3 verses).

    Returns
    -------
    float
        Cosine similarity score in [−1, 1].  Higher = more semantically
        relevant.  A threshold of ~0.50 is a reasonable starting point.
    """
    if not els_word.strip() or not context_text.strip():
        return 0.0

    if _is_english_text(els_word):
        word_emb    = _embed_english(els_word)
        context_emb = _embed_english(context_text)
    else:
        word_emb    = _embed(els_word)
        context_emb = _embed(context_text)
    return float(torch.dot(word_emb, context_emb).item())


def embed_batch(texts: list[str], batch_size: int = 32) -> "torch.Tensor":
    """
    Embed a list of Hebrew texts in mini-batches.

    Returns a 2-D tensor of shape ``(len(texts), hidden_dim)`` where every row
    is the L2-normalised mean-pool embedding for the corresponding text.

    Parameters
    ----------
    texts : list[str]
        Hebrew Unicode strings to embed.
    batch_size : int
        Number of texts to tokenise and forward in a single model call (default 32).

    Returns
    -------
    torch.Tensor  shape (N, hidden_dim), each row has L2-norm == 1.0
    """
    if not texts:
        return torch.empty(0)

    tokenizer, model, device = _load_model()
    all_embeddings: list[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.last_hidden_state: (B, seq_len, hidden_dim)
        hidden = outputs.last_hidden_state                            # (B, S, H)
        mask   = inputs["attention_mask"].unsqueeze(-1).float()      # (B, S, 1)
        summed = (hidden * mask).sum(dim=1)                          # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)                     # (B, 1)
        mean_emb = summed / counts                                   # (B, H)
        normed   = F.normalize(mean_emb, dim=-1)                     # (B, H)
        all_embeddings.append(normed.cpu())

    return torch.cat(all_embeddings, dim=0)   # (N, H)


def score_pairs_batch(
    pairs: list[tuple[str, str]],
    batch_size: int = 32,
) -> list[float]:
    """
    Score a list of ``(word, context)`` pairs in a single batched pass through HeBERT.

    Both words and contexts are embedded via :func:`embed_batch`, then cosine
    similarity is computed as a dot product of the L2-normalised vectors.

    Parameters
    ----------
    pairs :
        ``[(els_word, context_text), …]``
    batch_size :
        Mini-batch size passed to :func:`embed_batch` (default 32).

    Returns
    -------
    list[float]
        Cosine similarity score for each pair, in [−1, 1].
    """
    if not pairs:
        return []
    words    = [p[0] for p in pairs]
    contexts = [p[1] for p in pairs]
    word_embs = embed_batch(words,    batch_size)   # (N, H)
    ctx_embs  = embed_batch(contexts, batch_size)   # (N, H)
    scores = (word_embs * ctx_embs).sum(dim=-1)    # (N,)  — dot of unit vecs = cosine sim
    return scores.tolist()


def warm_up() -> None:
    """Pre-load both Hebrew and English BERT models so the first search call is not slow."""
    _load_model()
    _load_eng_model()
