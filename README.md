# CodeSearch-Ultra

**Autonomous Forensic Intelligence Node — Decadal Horizon Mapping, 2026–2036**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Four Pillars of Ultra-Intelligence](#the-four-pillars-of-ultra-intelligence)
3. [The Intelligence Dashboard](#the-intelligence-dashboard)
4. [Technical Deployment](#technical-deployment)
5. [Forensic Status](#forensic-status)
6. [Architecture](#architecture)
7. [CLI Reference](#cli-reference)

---

## Executive Summary

CodeSearch-Ultra is an **Autonomous Forensic Intelligence Node** engineered for decadal horizon mapping across the **2026–2036** threat window. It operates on a 1.94-million-letter dual-language corpus — Hebrew Tanakh + English KJV — scanning for Equidistant Letter Sequence (ELS) signatures that correlate with geopolitical inflection points.

**Primary Mission:** Unmask geopolitical "Mirages" — surface-level events that mask deeper structural transitions — through dual-epoch semantic analysis. A primary active case is the **2026 Islamabad Accord** pattern cluster: ELS tokens intersecting Hebrew year **תשפו** (5786 / 2026) in Prophetic books at consensus scores exceeding 0.45, suggesting the year marks a systemic reorganisation threshold, not an isolated diplomatic event.

The platform functions in three simultaneous modes:

- **Passive Archive Mode** — continuous ChromaDB HNSW vector indexing of every significant hit across all search runs (42,204 entries and growing)
- **Active Ingestion Mode** — `SignalIngestor` polls global headlinse every 4 hours, translates trigger terms to Hebrew, and fires ELS scans autonomously
- **Analyst Review Mode** — Streamlit intelligence dashboard with five specialist panels including the Network Hub and ELS Grid visualizer

The three **Systemic Decadal Anchors** that structure this platform's reference frame:

| Hebrew Token | Gregorian Year | Hebrew Year | Designation            |
| ------------ | -------------- | ----------- | ---------------------- |
| **תשפו**     | 2026           | 5786        | Near-Horizon Threshold |
| **תשצ**      | 2030           | 5790        | Mid-Range Gravity Well |
| **תשצו**     | 2036           | 5796        | Far-Horizon Terminus   |

Any ELS hit whose skip structure resolves to one of these year tokens is automatically classified as a **Systemic Decadal Anchor** and flagged in all reporting outputs.

---

## The Four Pillars of Ultra-Intelligence

### Pillar I — Real-Time Signal Ingestion

**Module:** `bot/manager.py` → `SignalIngestor`

The `SignalIngestor` class monitors four live geopolitical feeds via NewsAPI every **4 hours**:

- `Israel` · `Iran` · `Jerusalem` · `Tesla IPO`

Extracted English headline keywords are translated to Hebrew through the cache-first Translation Bridge (`data/translation_cache.json`) before being submitted to `UltraSearchEngine`. If any resulting hit crosses the **0.50 consensus threshold**, the cycle is classified as a `CRITICAL SIGNAL` event:

- A tagged git commit is executed regardless of `--dry-run` status
- Commit message format: `alert(CRITICAL SIGNAL): N hit(s) >0.50 consensus [UTC timestamp]`
- The briefing document (`docs/decadal_horizon.md`) is updated with full forensic detail

Trigger the ingestor manually:

```powershell
# Single cycle — fetch, translate, search, auto-commit if critical
NEWS_API_KEY=<your_key> uv run bot/manager.py --ingest

# Standard pipeline run against specific terms
uv run bot/manager.py --words "תשפו" "ברית" "שלום" --commit
```

The `NEWS_API_KEY` environment variable must be set for live ingestion. Without it, the fallback seed list (`Israel`, `Iran`, `Jerusalem`, `Tesla IPO`) is translated directly and used as search terms with no HTTP call.

---

### Pillar II — Network Hub (Graph Theory)

**Module:** `analysis/network.py` → `VerseYearNetwork` · `GraphAnalyzer`

**Dashboard Panel:** Network Hub

Every ELS match is ingested into a **bipartite handshake graph** where:

- **Blue nodes** = Verse identifiers (`verse:Genesis 1:1`)
- **Orange nodes** = Search token strings (`word:תשצ`)
- **Green nodes** = Hebrew year codes (`year:תשפו`, `year:תשצ`, `year:תשצו`)
- **Red node** = **תשצ** (2030 Wall) — the designated Gravity Well

Edges carry weights derived from the `ConsensusScore.consensus` value of each hit. The `GraphAnalyzer.render()` method produces a self-contained PyVis force-directed HTML file, embedded live into the dashboard via `st.components.v1.html`.

**Gravity Well Report** — `GraphAnalyzer.gravity_well_report()` isolates the **תשצ** node and computes:

- Raw degree (number of unique verses and tokens connected to 2030)
- Weighted degree (sum of consensus-weighted edge values)
- `is_dominant` flag: True when **תשצ** weighted degree exceeds all other year nodes combined

The 2030 Wall hypothesis: as more search runs accumulate in the archive, **תשצ** should exhibit anomalously high centrality relative to surrounding year tokens — a structural "gravity well" in the semantic graph where verse clusters from multiple thematic domains converge.

---

### Pillar III — Dual-Model Linguistic Jury

**Module:** `engine/search.py` → `ConsensusScore`

Every ELS match is evaluated by two independent BERT models operating in parallel:

| Model                    | Source                  | Register     | Role               |
| ------------------------ | ----------------------- | ------------ | ------------------ |
| `avichr/heBERT`          | Modern Hebrew corpus    | Contemporary | Primary scorer     |
| `onlplab/alephbert-base` | Biblical/Ancient Hebrew | Classical    | Verification layer |

The jury produces a `ConsensusScore` with seven fields:

```
hebert_score          — cosine similarity, HeBERT (modern register)
alephbert_score       — cosine similarity, AlephBERT (ancient register)
consensus             — geometric mean of both scores
is_significant        — consensus ≥ 0.30 (minimum report threshold)
is_decadal_anchor     — word is a Systemic Decadal Anchor token
is_verified           — |aleph − hebert| / max(hebert, ε) ≤ 0.20  (±20% band)
is_deep_archetypal_anchor — alephbert > hebert × 1.15
```

**Verified** hits (±20% inter-model deviation) indicate the semantic signature is stable across both temporal registers — modern and ancient. This is the platform's primary reliability filter.

**Deep Archetypal Anchor** hits — where AlephBERT exceeds HeBERT by more than 15% — indicate the term carries stronger resonance in the ancient biblical register than in modern usage. These are flagged as carrying an archaic semantic load likely invisible to contemporary readers but structurally embedded in the Masoretic text.

The minimum significance threshold for all reporting is **0.30 consensus**. CRITICAL SIGNAL classification threshold is **0.50**.

---

### Pillar IV — 2D Tactical Envelope (Grid Visualizer)

**Module:** `viz/grid_painter.py` → `SatelliteScanner` · `paint_grid()`

**Dashboard Panel:** ELS Grid

Every significant ELS match is renderable as a **2D matrix** — a rectangular grid of Hebrew letters where the primary search term appears as a vertical sequence at the target skip interval. The grid visualizer performs three layered analyses simultaneously:

**Primary Term** (orange `#FF6B35`) — the target ELS word at the identified skip.

**Crossword Terms** (teal `#4ECDC4`) — any secondary search terms that intersect the primary word's cells in the matrix.

**Satellite Terms** (amber `#F7DC6F`) — the `SatelliteScanner` scans all 8 compass directions from every cell occupied by the primary word, searching for 18 **TACTICAL_VOCABULARY** roots:

| Root    | Gloss | Root     | Gloss    |
| ------- | ----- | -------- | -------- |
| **אש**  | Fire  | **שלום** | Peace    |
| **דם**  | Blood | **ברית** | Covenant |
| **קץ**  | End   | **עם**   | People   |
| **מות** | Death | **ארץ**  | Land     |
| **חרב** | Sword | **משיח** | Messiah  |
| **מלך** | King  | **שמש**  | Sun      |
| **אל**  | God   | **קול**  | Voice    |
| **יום** | Day   | **עת**   | Time     |
| **אור** | Light | **חשך**  | Darkness |

Satellite detection is spatial — these roots need not appear at any fixed skip. They need only occur in the adjacent 2D neighbourhood of the primary hit. A primary ELS word appearing physically adjacent to **חרב** (Sword) and **קץ** (End) in the matrix is flagged for immediate analyst review, independent of its consensus score.

**Overlap** cells (purple `#9B59B6`) — positions shared by two or more classified layers.

---

## The Intelligence Dashboard

**Launch command:**

```powershell
uv run streamlit run viz/dashboard.py
```

The Streamlit dashboard provides five analyst panels:

| Panel             | Function                                                                                                    |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| **Watchlist**     | All archived hits (consensus ≥ 0.30) from ChromaDB, paginated. Cross-run semantic discovery.                |
| **Live Search**   | Interactive ELS search with dual-model scoring, real-time `st.progress()` feedback per word.                |
| **ELS Grid**      | 2D matrix visualizer with Primary / Crossword / Satellite / Overlap cell layers.                            |
| **Network Hub**   | PyVis force-directed graph, gravity well report (**תשצ** centrality metrics), betweenness centrality table. |
| **Chronos Audit** | Decadal Anchor cross-reference summary across all archived runs.                                            |

**Translation Bridge** (sidebar, all panels) — Analysts input English terms; the bridge performs cache-first translation to Hebrew via `deep_translator` (Google backend), appends the translated token to the active search set, and displays it inline with its English gloss (`גנטי (genetic)`). This allows English-language intent — sourced from news monitoring or analyst instinct — to be mapped directly to Hebrew ELS search without manual transliteration.

Search progress is streamed live: as each word's skip scan completes, the progress bar advances and partial results render immediately. A search across five Hebrew terms at max_skip=1,000 produces visible output within seconds of each word completing — no waiting for the full batch.

---

## Technical Deployment

### Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) package manager
- `NEWS_API_KEY` environment variable (for `SignalIngestor` live ingestion)

### Install

```powershell
git clone <repo>
cd CodeSearch
uv sync
```

### Launch Intelligence Dashboard

```powershell
uv run streamlit run viz/dashboard.py
```

### Run Platform Manager (CLI)

```powershell
# Standard pipeline — search, score, commit
uv run bot/manager.py --words "תשפו" "ברית" "שלום" --commit

# Dry run — search and score without committing
uv run bot/manager.py --words "משיח" "קץ" --dry-run

# Single SignalIngestor cycle — live news fetch + ELS search + auto-commit if critical
NEWS_API_KEY=<key> uv run bot/manager.py --ingest
```

### Rebuild Corpus

```powershell
uv run python backend/fetch_kjv_nt.py
Remove-Item data\full_text.bin, data\index_map.parquet -ErrorAction SilentlyContinue
uv run python backend/data_loader.py
```

Expected output:

```
Loaded 39 Tanakh books  (1,202,701 Hebrew letters)
Loaded 27 KJV NT books  (    739,965 English letters)
Total corpus: 1,942,666 letters
```

### Model Downloads (First Run)

HeBERT (`avichr/heBERT`, 438 MB) and AlephBERT (`onlplab/alephbert-base`) download automatically from Hugging Face on first `--validate` run. Both are cached at `~/.cache/huggingface/`.

---

## Forensic Status

**Archive Size:** 42,204 entries and growing. ChromaDB HNSW index (`data/chroma/`), cosine metric, Hebrew ELS matches only.

**Primary Active Watch — April 22, 2026 (5 Iyar תשפו):**
The "Hidden Wheat" transition date. ELS clusters intersecting **תשפו** in Prophetic texts show anomalous density around agricultural and covenant terminology. 5 Iyar is also Israeli Independence Day — a structural ambiguity in the corpus that the Linguistic Jury is currently resolving via `is_deep_archetypal_anchor` classification (ancient vs. modern semantic register).

**Systemic Decadal Anchor Status:**

| Token           | Status         | Notes                                                     |
| --------------- | -------------- | --------------------------------------------------------- |
| **תשפו** (2026) | ACTIVE WATCH   | High-density cluster in Neviim. Near-Horizon Threshold.   |
| **תשצ** (2030)  | GRAVITY WELL   | Designated network hub node. Centrality analysis ongoing. |
| **תשצו** (2036) | REMOTE HORIZON | Low current density. Expected to grow as 2030 resolves.   |

**Top Consensus Hit on Record:** Run 22 — נהמה + אריה (Roaring + Lion), Corpus Z = **+51.09 ★★★**, 1,264 hits archived.

---

## Architecture

### 1-Byte Encoding Firewall

The corpus is compacted into a single `bytes` object. Every letter occupies exactly one byte:

| Range | Content                                               | Values              |
| ----- | ----------------------------------------------------- | ------------------- |
| 1–27  | Hebrew consonants (alef → tav, final-form normalised) | 1 = alef … 27 = tav |
| 28–53 | English letters A–Z                                   | 28 = A … 53 = Z     |
| 0     | Gap                                                   | —                   |

A Hebrew search pattern (bytes 1–27) cannot match KJV English text (bytes 28–53). Cross-language contamination is structurally impossible.

### Dual-BERT Consensus Pipeline

```
UltraSearchEngine.run(words)
  └── ELSEngine.search_single(word)          → raw Match list (StringZilla SIMD)
        └── _build_consensus(match)
              ├── HeBERT cosine(word, verse)  → hebert_score
              ├── AlephBERT cosine(word, verse) → alephbert_score
              ├── deviation = |aleph - hebert| / max(hebert, ε)
              ├── is_verified = deviation ≤ 0.20
              ├── is_deep_archetypal_anchor = aleph > hebert × 1.15
              └── ConsensusScore(7 fields)
```

### ChromaDB Semantic Archive

Significant Hebrew matches (`consensus ≥ 0.30`) are stored with HeBERT word embeddings in a ChromaDB HNSW index. Cross-run semantic queries embed the query string and return ranked nearest neighbours — bridging research sessions without lexical overlap.

---

## CLI Reference

```powershell
uv run main.py --words WORD [WORD …] [OPTIONS]
```

| Flag                  | Default  | Description                                                                      |
| --------------------- | -------- | -------------------------------------------------------------------------------- |
| `--words`             | required | ELS search terms (Hebrew or English)                                             |
| `--books`             | all      | Book filter — name or group alias (`Torah`, `Tanakh`, `NT`, `Neviim`, `Ketuvim`) |
| `--min-skip N`        | 1        | Minimum absolute skip                                                            |
| `--max-skip N`        | 1000     | Maximum absolute skip                                                            |
| `--validate`          | off      | Run Dual-BERT consensus scoring                                                  |
| `--baseline`          | off      | Monte Carlo Z-score (100 shuffles)                                               |
| `--baseline-trials N` | 100      | Number of Monte Carlo shuffles                                                   |
| `--long-skip`         | off      | Discard `\|skip\| < 10` (isolate hidden patterns)                                |
| `--auto-scale`        | off      | Raise max_skip for words ≥ 5 letters                                             |
| `--translate`         | off      | Auto-translate English terms to Hebrew                                           |
| `--expand`            | off      | Add Hebrew synonyms from static lexicon                                          |
| `--archive`           | off      | Store significant hits in ChromaDB                                               |
| `--view-grid`         | off      | Generate 2D ELS matrix HTML                                                      |
| `--output FILE`       | —        | Export results to CSV                                                            |
| `--dry-run`           | off      | Parse and validate without writing output                                        |
