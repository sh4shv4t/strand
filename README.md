# Strand

[![CI](https://github.com/sh4shv4t/strand/actions/workflows/ci.yml/badge.svg)](https://github.com/sh4shv4t/strand/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-4338CA?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-10B981?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/frontend-React%2019-4338CA?logo=react&logoColor=white)
![Docker](https://img.shields.io/badge/deploy-Docker%20Compose-10B981?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-59%20passing-4338CA)

Compositional fashion image search — retrieval that binds garments, colors, scene, and style as separate fields instead of pooling everything into one embedding, so a query like "a red tie and a white shirt" doesn't also match a white tie and a red shirt.

Built for the Glance ML internship take-home assignment. See [`Working_notes.md`](./Working_notes.md) for the full engineering log: architecture options considered, dataset plan, empirical results, and open decisions.

![Strand search UI showing the compositional query "A red tie and a white shirt in a formal setting", with the true match ranked first at 93% and its color-swapped decoy second at 53%](./assets/screenshot.png)

## Overview

Vanilla CLIP/SigLIP embeddings encode a whole image as one dense vector, which is enough for coarse retrieval but fails on compositional queries — attributes get "bag of words"-ed together, so color-swapped images score similarly. Strand instead extracts a structured schema per image (garments bound to slots, plus scene and style) and scores queries against that schema with a weighted-hybrid retriever, falling back to dense similarity for phrasing the schema doesn't capture.

The catalog is 1,000 real Fashionpedia photos (garment detection from the dataset's own ground truth) plus 12 hand-written records that isolate the compositional-binding failure with a color-swapped decoy pair. Retrieval blends three real signals: symbolic slot/type/color matching, caption-text similarity, and real CLIP image-pixel similarity, the last of which is compared against an actual persisted embedding per photo, not a stand-in. The query parser is a two-tier real-LLM-first design (`services/query_parsing.py`): a real Gemini parser when `GEMINI_API_KEY` is set, transparently falling back to a rule-based keyword parser otherwise, so the app always answers rather than erroring. `color`, `scene`, and `style` on the real photos are still `null` pending that same key, see [Status and limitations](#status-and-limitations).

### Part A / Part B mapping

The assignment asks for two distinct workflows. Strand keeps them as one FastAPI app with a module-level split rather than two top-level directories (see `Working_notes.md` §9 for why), so here is the explicit mapping:

| | Files | Entry points |
|---|---|---|
| **Part A: The Indexer** | `services/indexer.py`, `services/clip_model.py`, `services/image_vector_store.py`, `services/vlm_attribute_extractor.py` | `POST /api/index`, `scripts/build_image_vector_index.py`, `scripts/extract_attributes_with_vlm.py` |
| **Part B: The Retriever** | `services/query_parsing.py`, `services/llm_query_parser.py`, `services/query_parser.py`, `services/retriever.py`, `services/vector_store.py`, `services/image_similarity.py`, `services/garment_vocabulary.py` | `POST /api/query`, `scripts/search.py` |
| **Shared** | `schema.py` (both sides speak the same `ExtractedAttributes`/`ImageRecord` schema), `services/catalog.py` | — |

## Architecture

```mermaid
flowchart LR
  subgraph Browser
    UI["React app"]
  end
  subgraph Frontend["nginx (prod) / Vite dev server"]
    PX["reverse proxy: /api/* → backend"]
  end
  subgraph Backend["FastAPI"]
    MW["observability middleware"]
    RQ["/api/query"]
    RC["/api/catalog"]
    RI["/api/index (Part A,<br/>+ optional cold-start registration)"]
    RH["/api/health"]
    RIM["/api/images/*.jpg"]
  end
  subgraph Data
    CAT[("catalog JSON<br/>mock + 1,000 real Fashionpedia")]
    CH[("Chroma<br/>caption embeddings")]
    IMG[("real photos<br/>on disk")]
    IVX[("Chroma<br/>real CLIP image embeddings")]
  end
  UI --> PX --> MW
  MW --> RQ & RC & RI & RH & RIM
  RQ --> CAT
  RQ --> CH
  RQ --> IVX
  RC --> CAT
  RI --> IVX
  RI -.->|"registers a new record"| CAT
  RIM --> IMG
```

Same-origin from the browser's point of view: nginx proxies `/api/*` server-side in production, so no CORS handling is needed on that path at all. `/api/query` reads both Chroma collections, real image embeddings and caption embeddings are two independent signals blended together, see the query pipeline below.

### Query pipeline

```mermaid
flowchart LR
  Q["NL query"] --> P["query_parsing.py<br/>real LLM parser, falls back to<br/>keyword-spotting if no API key"]
  P --> PS["ParsedQuery<br/>garments + scene + style + confidence"]
  PS --> SYM["Symbolic score<br/>slot + canonical-type + color match"]
  PS --> CAP["Caption dense score<br/>Chroma cosine similarity"]
  PS --> IMGSIM["Image dense score<br/>real CLIP text↔image cosine similarity"]
  CAP --> BLEND2["dense = mean(caption, image)<br/>caption-only if no stored embedding"]
  IMGSIM --> BLEND2
  SYM --> BLEND["score = α·symbolic + (1−α)·dense<br/>α scaled by parse confidence,<br/>0 if nothing was recognized"]
  BLEND2 --> BLEND
  BLEND --> RANK["Ranked results"]
```

The query path only ever touches the lightweight parser, an LLM call (if configured), and approximate nearest-neighbor lookups, never a heavy model retrained at request time, which is what makes the scalability argument in `Working_notes.md` §13/§13.1 hold at any catalog size.

## Tech stack

- **Backend:** Python, FastAPI, Pydantic
- **Frontend:** React, TypeScript, Vite, Tailwind CSS

## Project structure

```
strand/
├── Working_notes.md      # architecture options, dataset plan, tradeoffs, open decisions
├── .github/workflows/ci.yml   # backend pytest + ruff + frontend build/lint on push/PR
├── docker-compose.yml     # backend + frontend, wired together
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml     # ruff config
│   ├── .env.example       # env vars, documented; sane defaults without a .env at all
│   ├── scripts/
│   │   ├── search.py                     # plain CLI: python scripts/search.py "a query"
│   │   ├── pull_fashionpedia_sample.py   # regenerates real_catalog_sample.json + images (1,000)
│   │   ├── eval_baselines.py             # Tier 1: dense-only vs. hybrid comparison
│   │   ├── eval_clip_baseline.py         # Tier 2: real vanilla-CLIP image baseline
│   │   ├── tag_real_catalog_scene_style.py   # zero-shot CLIP tagging, tried and not applied, see Working_notes.md §12.3
│   │   ├── tune_alpha.py                 # empirical alpha sweep, see Working_notes.md §8
│   │   ├── build_image_vector_index.py   # Part A batch driver: real CLIP feature extraction + persistent storage
│   │   └── extract_attributes_with_vlm.py   # fills in color/scene/style via Gemini, needs GEMINI_API_KEY
│   ├── tests/                  # pytest suite, see Testing below
│   └── app/
│       ├── schema.py          # Pydantic models, incl. the shared ExtractedAttributes schema
│       ├── observability.py   # structured logging + OpenTelemetry (console exporter)
│       ├── data/               # mock decoy-pair records, the real Fashionpedia sample, real photos
│       ├── services/
│       │   ├── query_parsing.py       # entry point: real LLM parser, falls back to keywords
│       │   ├── query_parser.py        # keyword-spotting fallback parser
│       │   ├── llm_query_parser.py    # real Gemini query parser
│       │   ├── vlm_attribute_extractor.py  # real Gemini image attribute extractor
│       │   ├── gemini_client.py       # shared Gemini SDK wrapper
│       │   ├── catalog.py         # loads/caches the combined catalog
│       │   ├── vector_store.py    # caption dense similarity via a local Chroma collection
│       │   ├── image_similarity.py    # real CLIP image-pixel similarity, query text vs. stored embeddings
│       │   ├── garment_vocabulary.py   # curated garment-type synonym canonicalization
│       │   ├── retriever.py       # weighted-hybrid scoring (symbolic + blended dense)
│       │   ├── clip_model.py      # shared local CLIP model, used by indexer.py and image_similarity.py
│       │   ├── indexer.py         # Part A: real CLIP feature extraction, no API key needed
│       │   └── image_vector_store.py  # persistent Chroma collection for real image embeddings
│       └── routers/            # /api/query, /api/catalog, /api/index
└── frontend/
    └── src/
        ├── components/          # SearchBar, ExampleChips, ResultCard, Logo
        ├── lib/api.ts           # typed client for the query endpoint
        └── App.tsx
```

## Getting started

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

This alone gets you a working system: the real Fashionpedia garment/scene/style data is already in the committed `real_catalog_sample.json`, so symbolic and caption-dense scoring both work immediately. Two things are gitignored and regenerable rather than committed (101MB of JPEGs and a CLIP embedding index don't belong in git), and without them the app still runs and answers queries correctly, but degrades quietly instead of erroring:

- **The real photos** (`app/data/fashionpedia_images/`): without them, `/api/images/*.jpg` 404s and the UI shows broken image links instead of real photos.
- **The real CLIP image embeddings** (`app/data/image_vector_index/`): without them, `image_similarity.score()` returns `{}` for every query and ranking silently falls back to caption-only dense similarity, still correct, just missing the image-pixel signal `Working_notes.md` §14.1 describes.

To get both:

```bash
pip install -r scripts/requirements-eval.txt   # adds datasets, pillow, open_clip_torch
python scripts/pull_fashionpedia_sample.py     # downloads the 1,000 real photos
python scripts/build_image_vector_index.py     # embeds them with local CLIP, no API key
```

Each is independently re-runnable and idempotent (safe to run again after a fresh clone or a dependency bump).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server proxies `/api/*` to `http://localhost:8000`. Open the printed local URL (default `http://localhost:5173`).

### Docker

```bash
docker compose up --build
```

Serves the frontend at `http://localhost:5173` (nginx, proxying `/api/*` server-side to the backend container — no CORS involved) and the backend directly at `http://localhost:8000`. First boot downloads the ~80MB Chroma embedding model before the backend responds to anything; a named volume (`chroma-cache`) persists it so this only happens once. Copy `backend/.env.example` to `backend/.env` and uncomment the `env_file` line in `docker-compose.yml` to pass real env vars through.

Same caveat as above applies here: `backend/Dockerfile` does a one-time `COPY app ./app` at build time, so run the two `pull_fashionpedia_sample.py` / `build_image_vector_index.py` commands above *before* `docker compose up --build`, not after, and rebuild whenever you regenerate them. Without that, the container starts and serves correct results, just without real photos or the image-similarity signal, same degrade as running the backend directly.

## Search from the command line

```bash
cd backend
python scripts/search.py "a red tie and a white shirt in a formal setting"
```

The literal Part B requirement, "a script that accepts a natural language string and returns the top k matching images", as a plain one-line CLI, no server or browser needed. Uses the exact same code path the API uses (`query_parsing.parse_query` + `retriever.search`), so it automatically uses the real LLM parser once `GEMINI_API_KEY` is set, and the same keyword-spotting fallback otherwise, no separate implementation to drift out of sync. `--top-k` and `--alpha` are optional.

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
ruff check .
pytest -v
```

Tests run with `STRAND_DISABLE_EMBEDDINGS=1` (set in `tests/conftest.py`) so they exercise the deterministic word-overlap fallback instead of downloading the real embedding model — fast and network-independent. `tests/test_eval_accuracy.py` runs the 5 canonical eval queries from `Working_notes.md` end to end and checks retrieval accuracy, not just unit-level correctness. CI (`.github/workflows/ci.yml`) runs `ruff check`, this pytest suite, and a frontend build/lint on every push and PR.

## Baseline comparison

```bash
cd backend
python scripts/eval_baselines.py                      # dense-only vs. hybrid, no new deps
pip install -r scripts/requirements-eval.txt
python scripts/eval_clip_baseline.py                   # real vanilla-CLIP image baseline
```

Real numbers, not just theory: dense-only ties the hybrid on the 5 curated eval queries, but produces an *exact* score tie on the compositional decoy pair (it truly cannot tell "red tie, white shirt" from "white tie, red shirt" apart). On the real 1,000-photo catalog, real vanilla CLIP, our dense-only fallback, and our hybrid score 0.725 / 0.850 / **1.000** mean precision@5 across 8 single-garment probe queries against Fashionpedia's own ground truth. See `Working_notes.md` §12 for the full breakdown, why precision@5 replaced recall@5 as the meaningful metric once the catalog grew past a few hundred images, and the caveats.

## Part A: real feature extraction

```bash
cd backend
pip install -r scripts/requirements-eval.txt
python scripts/build_image_vector_index.py
```

Runs every real Fashionpedia photo on disk through a local CLIP model (`open_clip`, no API key) and persists a real embedding per image to an on-disk Chroma collection at `app/data/image_vector_index/` (gitignored, regenerable). This is genuine feature extraction from pixels, not from labels, closing the literal Part A requirement the mocked catalog alone doesn't. It doesn't touch color, scene, or style; that axis needs a real VLM call, see below. These embeddings aren't just persisted, `/api/query` actually reads them back at query time (`services/image_similarity.py`), encoding the query through the same model's text tower so it lands in the same joint space, real image-pixel similarity blended into ranking, not a stored-and-forgotten side effect.

## Real LLM and VLM integration (needs a Gemini API key)

```bash
cd backend
# GEMINI_API_KEY=... in backend/.env, see .env.example
python scripts/extract_attributes_with_vlm.py   # fills in color/scene/style on the real catalog
```

Two real Gemini call sites, both code-complete and unit-tested, neither exercised against a live key yet:

- `services/llm_query_parser.py` replaces the keyword-spotting parser at query time, using the exact prompt drafted in `Working_notes.md` §4.3.1.
- `services/vlm_attribute_extractor.py` extracts color, scene, and style from a real photo at index time, keeping Fashionpedia's own ground-truth garment presence and only asking the VLM for the axis it doesn't cover.

`services/query_parsing.py` is what `/api/query` actually calls: it tries the real parser first and falls back to the keyword parser automatically on `GeminiNotConfigured` or any other failure (network, rate limit, malformed response), so the app behaves exactly as it does today until a key is set, and degrades to that same behavior rather than erroring if a call ever fails afterward.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/query` | Parses a natural-language query and returns ranked, scored matches from the catalog |
| `GET` | `/api/catalog` | Returns the full indexed catalog |
| `POST` | `/api/index` | Part A: real CLIP feature extraction for one image. 200 with the embedding dimension on success, 404 if the file doesn't exist, 503 if `torch`/`open_clip` aren't installed in this deployment. Optionally accepts `garments`/`scene`/`style`, which additionally registers the image into the live catalog so it's immediately searchable (the cold-start fix, see `Working_notes.md` §14.3) |

## Status and limitations

Partway from mocked to real, not a finished system:

- The catalog is 1,000 real Fashionpedia records plus 12 hand-written mock records. Garment detection on the real records uses the dataset's own ground-truth labels, but `color`, `scene`, and `style` stay `null` there until `scripts/extract_attributes_with_vlm.py` is actually run with a key.
- The real LLM query parser and VLM attribute extractor are written and unit-tested (mocked/error-path tests only, no real API call made in CI) but not yet exercised against a live Gemini key. Until a key is set, `/api/query` transparently uses the keyword-spotting fallback, same as before this was added.
- Dense retrieval blends two real signals (caption-text similarity and real CLIP image-pixel similarity); symbolic retrieval (with garment-type synonym canonicalization) and the weighted-hybrid blend are real. Real image feature extraction (Part A) is implemented via local CLIP and is now actually consulted at query time, not just persisted.
- Constructing the caption-embedding index (`DenseScorer`) at process startup takes roughly 20 seconds against the full 1,000+12 record catalog, measured directly, a one-time cost paid once per process (server startup or CLI invocation), not per query; each query after that is well under a second. See `Working_notes.md` §14 for the measurement.

See `Working_notes.md` sections 9 through 14 for the full list of decisions made, what's still open, the testing/CI/observability setup, the empirical baseline comparison, and a measured scaling estimate.
