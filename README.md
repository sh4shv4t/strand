# Strand

Compositional fashion image search — retrieval that binds garments, colors, scene, and style as separate fields instead of pooling everything into one embedding, so a query like "a red tie and a white shirt" doesn't also match a white tie and a red shirt.

Built for the Glance ML internship take-home assignment. See [`Working_notes.md`](./Working_notes.md) for the full problem writeup: architecture options considered, dataset plan, and open decisions (the final architecture is intentionally not locked in yet).

## Overview

Vanilla CLIP/SigLIP embeddings encode a whole image as one dense vector, which is enough for coarse retrieval but fails on compositional queries — attributes get "bag of words"-ed together, so color-swapped images score similarly. Strand instead extracts a structured schema per image (garments bound to slots, plus scene and style) and scores queries against that schema with a weighted-hybrid retriever, falling back to dense similarity for phrasing the schema doesn't capture.

This repo currently ships a scaffold of that pipeline with mocked data: a hand-written catalog, a rule-based query parser standing in for an LLM parser, and a real implementation of the weighted-hybrid scoring, so the retrieval logic and API/UI contract are testable end-to-end ahead of real model and dataset integration.

## Tech stack

- **Backend:** Python, FastAPI, Pydantic
- **Frontend:** React, TypeScript, Vite, Tailwind CSS

## Project structure

```
strand/
├── Working_notes.md      # architecture options, dataset plan, tradeoffs, open decisions
├── .github/workflows/ci.yml   # backend pytest + frontend build/lint on push/PR
├── backend/
│   ├── scripts/
│   │   └── pull_fashionpedia_sample.py   # regenerates real_catalog_sample.json
│   ├── tests/                  # pytest suite, see Testing below
│   └── app/
│       ├── schema.py          # Pydantic models for the garment/scene/style JSON schema
│       ├── observability.py   # structured logging + OpenTelemetry (console exporter)
│       ├── data/               # mock decoy-pair records + a real Fashionpedia sample
│       ├── services/
│       │   ├── query_parser.py    # rule-based parser (stand-in for an LLM query parser)
│       │   ├── catalog.py         # loads/caches the combined catalog
│       │   ├── vector_store.py    # dense similarity via a local Chroma collection
│       │   ├── retriever.py       # weighted-hybrid scoring (symbolic + dense)
│       │   └── indexer.py         # stub documenting the real offline indexing pipeline
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

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server proxies `/api/*` to `http://localhost:8000`. Open the printed local URL (default `http://localhost:5173`).

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest -v
```

Tests run with `STRAND_DISABLE_EMBEDDINGS=1` (set in `tests/conftest.py`) so they exercise the deterministic word-overlap fallback instead of downloading the real embedding model — fast and network-independent. `tests/test_eval_accuracy.py` runs the 5 canonical eval queries from `Working_notes.md` end to end and checks retrieval accuracy, not just unit-level correctness. CI (`.github/workflows/ci.yml`) runs this suite plus a frontend build/lint on every push and PR.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/query` | Parses a natural-language query and returns ranked, scored matches from the catalog |
| `GET` | `/api/catalog` | Returns the full indexed catalog |
| `POST` | `/api/index` | Stub for the offline indexing pipeline — returns 501, not yet implemented |

## Status and limitations

Partway from mocked to real, not a finished system:

- The catalog combines 12 hand-written mock records with 40 real Fashionpedia records; garment detection on the real records uses the dataset's own labels, but `color`, `scene`, and `style` are honestly `null` there, not guessed.
- The query parser is keyword-spotting, not an LLM — a prompt to replace it is already drafted in `Working_notes.md` §4.3.1.
- Dense retrieval uses real embeddings (Chroma, local, no API key); symbolic retrieval and the weighted-hybrid blend are real. No VLM is integrated yet, so the fields it would fill in stay empty.

See `Working_notes.md` sections 9–11 for the full list of decisions made, what's still open, and the testing/CI/observability setup.
