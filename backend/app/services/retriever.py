"""Weighted-hybrid retrieval scoring.

Implements the `score = alpha * symbolic_match + (1 - alpha) * dense_cosine`
formula from Working_notes.md Section 3, against the catalog (mock
decoy-pair records + a real Fashionpedia sample, see app/data/).

The dense half uses a local Chroma collection (default embedding function,
no API key) for real cosine similarity between the query and each record's
flattened caption. If the embedding model can't be loaded (e.g. no network
on first run, before its weights are cached), this falls back to word
overlap between the query and caption -- degraded, but the app still
starts and answers queries rather than failing outright.
"""

import json
import re
from importlib import resources

import chromadb

from app.schema import Garment, ImageRecord, ParsedQuery, ScoredResult


def _load_records(filename: str) -> list[ImageRecord]:
    data_path = resources.files("app.data").joinpath(filename)
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return [ImageRecord(**item) for item in raw]


_CATALOG: list[ImageRecord] = _load_records("sample_catalog.json") + _load_records(
    "real_catalog_sample.json"
)


def get_catalog() -> list[ImageRecord]:
    return _CATALOG


def _build_chroma_collection():
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            "strand_catalog", metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            ids=[r.id for r in _CATALOG],
            documents=[r.caption for r in _CATALOG],
        )
        return collection
    except Exception:
        return None


_COLLECTION = _build_chroma_collection()

_WORD_RE = re.compile(r"\w+")


def _word_overlap_score(raw_query: str, record: ImageRecord) -> float:
    query_words = set(_WORD_RE.findall(raw_query.lower()))
    caption_words = set(_WORD_RE.findall(record.caption.lower()))
    if not query_words or not caption_words:
        return 0.0
    overlap = query_words & caption_words
    return len(overlap) / len(query_words | caption_words)


def _dense_scores(raw_query: str) -> dict[str, float]:
    if _COLLECTION is not None:
        try:
            result = _COLLECTION.query(query_texts=[raw_query], n_results=len(_CATALOG))
            ids = result["ids"][0]
            distances = result["distances"][0]
            # cosine space: distance = 1 - cosine_similarity
            return {rid: max(0.0, min(1.0, 1.0 - dist)) for rid, dist in zip(ids, distances)}
        except Exception:
            pass
    return {record.id: _word_overlap_score(raw_query, record) for record in _CATALOG}


def _garment_matches(query_garment: Garment, record_garments: list[Garment]) -> bool:
    for rg in record_garments:
        if rg.slot != query_garment.slot:
            continue
        if query_garment.type and query_garment.type != rg.type:
            continue
        if query_garment.color and rg.color and query_garment.color != rg.color:
            continue
        return True
    return False


def _symbolic_score(parsed: ParsedQuery, record: ImageRecord) -> tuple[float, list[str]]:
    total_fields = len(parsed.garments) + (1 if parsed.scene else 0) + (1 if parsed.style else 0)
    if total_fields == 0:
        return 0.0, []

    matched_fields: list[str] = []

    for g in parsed.garments:
        if _garment_matches(g, record.garments):
            label = f"{g.color + ' ' if g.color else ''}{g.type}".strip()
            matched_fields.append(label)

    if parsed.scene and parsed.scene == record.scene:
        matched_fields.append(f"scene:{parsed.scene}")

    if parsed.style and parsed.style == record.style:
        matched_fields.append(f"style:{parsed.style}")

    matched_count = (
        sum(1 for g in parsed.garments if _garment_matches(g, record.garments))
        + (1 if parsed.scene and parsed.scene == record.scene else 0)
        + (1 if parsed.style and parsed.style == record.style else 0)
    )

    return matched_count / total_fields, matched_fields


def search(parsed: ParsedQuery, top_k: int = 5, alpha: float = 0.6) -> list[ScoredResult]:
    dense_scores = _dense_scores(parsed.raw_query)
    scored: list[ScoredResult] = []

    for record in _CATALOG:
        symbolic, matched_fields = _symbolic_score(parsed, record)
        dense = dense_scores.get(record.id, 0.0)
        score = alpha * symbolic + (1 - alpha) * dense

        scored.append(
            ScoredResult(
                record=record,
                score=round(score, 4),
                symbolic_score=round(symbolic, 4),
                dense_score=round(dense, 4),
                matched_fields=matched_fields,
            )
        )

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
