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
├── backend/
│   └── app/
│       ├── schema.py          # Pydantic models for the garment/scene/style JSON schema
│       ├── data/               # mock indexed catalog (stand-in for a real dataset index)
│       ├── services/
│       │   ├── query_parser.py    # rule-based parser (stand-in for an LLM query parser)
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

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/query` | Parses a natural-language query and returns ranked, scored matches from the catalog |
| `GET` | `/api/catalog` | Returns the full indexed catalog |
| `POST` | `/api/index` | Stub for the offline indexing pipeline — returns 501, not yet implemented |

## Status and limitations

This is an initial scaffold, not a finished system:

- The catalog is 12 hand-written mock records, not a real indexed dataset.
- The query parser is keyword-spotting, not an LLM.
- No embedding model, VLM, or vector database is integrated yet.

See `Working_notes.md` sections 9 and 10 for the full list of open decisions and what a production implementation would still need.
