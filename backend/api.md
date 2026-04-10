# CodeSearch API — Technical Specification

Version: `0.1.0`  
Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Starting the server

```powershell
# Development (hot-reload)
uv run fastapi dev backend/api.py

# Production
uv run uvicorn backend.api:app --port 8000
```

The server loads the full 1.94M-letter corpus on first request and caches the `ELSEngine` instance in-process. Subsequent requests on the same `(min_skip, max_skip, validate)` triple reuse the cached engine with no re-parsing cost.

---

## Endpoints

### `POST /search` — ELS Discovery

Run an equidistant letter sequence search and return matches with statistical and semantic scores.

#### Request body (`application/json`)

```json
{
  "words": ["MESSIAH"],
  "books": ["Matthew", "John", "Acts"],
  "min_skip": 1,
  "max_skip": 50000,
  "run_validate": true,
  "top": 10
}
```

| Field          | Type               | Default            | Description                                                                                                                                                                          |
| -------------- | ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `words`        | `string[]`         | required           | ELS search terms. Hebrew consonants (Unicode U+05D0–U+05EA) or uppercase ASCII. Mixed-language arrays are valid — each word is independently routed to the correct corpus segment.   |
| `books`        | `string[] \| null` | `null` (all books) | Book name filter. Accepts individual book names (`"Genesis"`, `"John"`) or group aliases (`"Tanakh"`, `"NT"`, `"Neviim"`, `"Torah"`, `"Ketuvim"`). `null` searches the full corpus.  |
| `min_skip`     | `integer ≥ 1`      | `1`                | Minimum absolute ELS skip distance. Set to `10` to replicate `--long-skip` behaviour (hidden patterns only).                                                                         |
| `max_skip`     | `integer ≥ 1`      | `1000`             | Maximum absolute ELS skip distance. For 5+ letter words set to `10000`–`50000` to compensate for low hit probability.                                                                |
| `run_validate` | `boolean`          | `false`            | Score each match with the Dual-BERT pipeline. Hebrew words use `avichr/heBERT`; English words use `sentence-transformers/all-mpnet-base-v2`. Adds latency (~5–10 s per 100 matches). |
| `top`          | `integer ≥ 0`      | `0`                | If > 0, return only the top-N matches sorted by `\|skip\|` ascending. `0` = return all.                                                                                              |

#### Response body (`application/json`)

```json
{
  "count": 1,
  "wall_secs": 18.4,
  "matches": [
    {
      "word": "MESSIAH",
      "book": "John",
      "verse": "John 14:23",
      "skip": 7744,
      "start": 1517824,
      "sequence": "MESSIAH",
      "hebert_score": 0.3588,
      "z_score": 4.34,
      "is_significant": true
    }
  ]
}
```

| Field                      | Type           | Description                                                                                                                                                                                                                            |
| -------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `count`                    | `integer`      | Total number of matches returned.                                                                                                                                                                                                      |
| `wall_secs`                | `float`        | Wall-clock search time in seconds.                                                                                                                                                                                                     |
| `matches[].word`           | `string`       | The search term that produced this match.                                                                                                                                                                                              |
| `matches[].book`           | `string`       | Canonical book name (e.g. `"John"`, `"Isaiah"`, `"I Chronicles"`).                                                                                                                                                                     |
| `matches[].verse`          | `string`       | Verse reference of the anchor letter (the first letter of the ELS sequence).                                                                                                                                                           |
| `matches[].skip`           | `integer`      | ELS skip distance. Positive = left-to-right; negative = right-to-left.                                                                                                                                                                 |
| `matches[].start`          | `integer`      | Zero-based global corpus index of the first letter.                                                                                                                                                                                    |
| `matches[].sequence`       | `string`       | The actual letters decoded from the corpus at the ELS positions.                                                                                                                                                                       |
| `matches[].hebert_score`   | `float [0, 1]` | Cosine similarity between the ELS word embedding and the surrounding verse-context embedding. `0.0` when `run_validate=false`. For KJV NT books, this is an English BERT score; for Tanakh books, it is a HeBERT score.                |
| `matches[].z_score`        | `float`        | Corpus-level Z-score: `(real_hits − μ_null) / σ_null`. `0.0` unless `run_validate=true` **and** a baseline was computed server-side. Note: the `/search` endpoint does not run Monte Carlo by default — for full Z-scores use the CLI. |
| `matches[].is_significant` | `boolean`      | `true` when `z_score ≥ 3.0`.                                                                                                                                                                                                           |

#### Corpus Z-score vs Semantic Z-score

The REST `/search` endpoint does not perform Monte Carlo baseline testing (it would add 60–300 s per request). `z_score` and `is_significant` will be `0` / `false` unless the `ELSEngine` was pre-loaded with pre-computed baseline data.

For **Corpus Z-score** and **Semantic Z-score (SemZ)**, use the CLI with `--baseline --baseline-trials 100` and `--sem-sample 200`.

#### Language routing

| Input                   | Corpus segment searched     | Validator                                 |
| ----------------------- | --------------------------- | ----------------------------------------- |
| Hebrew (U+05D0–U+05EA)  | Tanakh (bytes 1–27)         | `avichr/heBERT`                           |
| English uppercase ASCII | KJV NT (bytes 28–53)        | `sentence-transformers/all-mpnet-base-v2` |
| Mixed array             | Both segments independently | Per-word routing                          |

---

### `GET /discover` — Semantic Cluster Lookup

Query the ChromaDB HNSW archive for ELS matches thematically similar to a query phrase. Does not require keyword overlap — uses HeBERT vector similarity.

#### Query parameters

| Parameter   | Type              | Default  | Description                                                                                                                    |
| ----------- | ----------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `query`     | `string`          | required | Hebrew or English theme phrase. Embedded with HeBERT; nearest neighbours retrieved from the `bible_codes` ChromaDB collection. |
| `n_results` | `integer [1, 50]` | `5`      | Number of nearest neighbours to return.                                                                                        |

#### Example

```
GET /discover?query=Deliverance%20and%20Restoration&n_results=5
```

#### Response body

```json
{
  "count": 5,
  "results": [
    {
      "id": "a3f2c1d4...",
      "word": "שלוה",
      "book": "Exodus",
      "verse": "Exodus 72:61",
      "skip": -1,
      "z_score": 14.98,
      "hebert_score": 0.3084,
      "semantic_z_score": -8.21,
      "is_significant": true,
      "start": 123456,
      "length": 4,
      "original_text": "שלוה",
      "distance": 0.5493
    }
  ]
}
```

| Field                     | Type           | Description                                                                     |
| ------------------------- | -------------- | ------------------------------------------------------------------------------- |
| `results[].distance`      | `float [0, 1]` | ChromaDB cosine distance (lower = more similar). `0.0` = exact embedding match. |
| `results[].original_text` | `string`       | The ELS letter sequence as decoded from the corpus (= `sequence` in `/search`). |
| All other fields          | —              | Same fields as `matches[]` in `/search`.                                        |

**Semantic centroid note:**  
After Runs 20–23 the archive contains ~5,490 entries. Queries about divine judgment, covenant, or restoration consistently return **ציונ** (Zion) as the nearest-neighbour attractor, reflecting HeBERT's Hebrew embedding geometry placing Zion at the intersection of redemption and judgment clusters.

---

### `GET /db-stats` — Archive Statistics

Return the current ChromaDB collection size and storage path.

#### Response body

```json
{
  "count": 5490,
  "collection": "bible_codes",
  "db_dir": "C:\\Users\\...\\CodeSearch\\data\\chroma"
}
```

| Field        | Type      | Description                                             |
| ------------ | --------- | ------------------------------------------------------- |
| `count`      | `integer` | Total number of archived ELS entries.                   |
| `collection` | `string`  | ChromaDB collection name (always `"bible_codes"`).      |
| `db_dir`     | `string`  | Absolute path to the ChromaDB SQLite storage directory. |

---

### `GET /` — Research UI

Serves `frontend/index.html` — the static Google Stitch research cockpit. If the `frontend/` directory is absent this route is not mounted (API-only mode).

---

## Error responses

All endpoints return standard FastAPI error envelopes on failure:

```json
{
  "detail": "string describing the error"
}
```

| Status                      | Cause                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `422 Unprocessable Entity`  | Request body/params failed Pydantic validation (wrong types, missing required field, out-of-range value). |
| `500 Internal Server Error` | Engine error — corpus files missing, ChromaDB unavailable, model loading failure.                         |

---

## Architecture notes

### Engine singleton

`_get_engine(min_skip, max_skip, validate)` maintains a single `ELSEngine` instance across requests. The engine is rebuilt only when these three parameters change. This means:

- First request: ~0.5 s corpus load + optional model warm-up (~2 s if `run_validate=true`)
- Subsequent requests with the same params: **no re-parse cost**
- Parameter change: engine is rebuilt on the next request

### Thread model

`uvicorn` spawns one async worker per CPU by default. The `ELSEngine` uses `concurrent.futures.ThreadPoolExecutor` internally for skip-range parallelism. For `--threads auto`: 1 thread for narrow searches (≤2 words, skip range ≤ 20,000); all logical CPUs otherwise.

### ChromaDB availability

`/discover` and `/db-stats` require `data/chroma/` to exist and contain at least one archived run. If the archive is empty or absent:

- `/db-stats` returns `{"count": 0, ...}`
- `/discover` returns an empty results list (or a 500 if ChromaDB cannot initialise)

Populate the archive by running any CLI search with `--validate --baseline --archive`.

### Language separation in the archive

The ChromaDB collection stores **Hebrew-only** ELS matches. KJV NT matches are excluded by the archiver gate (`book not in _KJV_NT_BOOK_NAMES`). This preserves a linguistically clean vector space: all stored embeddings are from `avichr/heBERT`, so cosine distances are metrically consistent.
