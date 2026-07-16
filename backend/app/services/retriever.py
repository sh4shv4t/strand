"""Weighted-hybrid retrieval scoring.

Implements the `score = alpha * symbolic_match + (1 - alpha) * dense_cosine`
formula from Working_notes.md Section 3, against the mock catalog.

The "dense" half is a stand-in: real word-overlap between the query and
each record's flattened caption, in place of cosine similarity between
real embeddings from a fashion-tuned encoder. Swapping in a real encoder
later only changes `_dense_score`; the blending logic and API stay the
same.
"""

import json
import re
from importlib import resources

from app.schema import Garment, ImageRecord, ParsedQuery, ScoredResult


def _load_catalog() -> list[ImageRecord]:
    data_path = resources.files("app.data").joinpath("sample_catalog.json")
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return [ImageRecord(**item) for item in raw]


_CATALOG = _load_catalog()


def get_catalog() -> list[ImageRecord]:
    return _CATALOG


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


_WORD_RE = re.compile(r"\w+")


def _dense_score(raw_query: str, record: ImageRecord) -> float:
    query_words = set(_WORD_RE.findall(raw_query.lower()))
    caption_words = set(_WORD_RE.findall(record.caption.lower()))
    if not query_words or not caption_words:
        return 0.0
    overlap = query_words & caption_words
    return len(overlap) / len(query_words | caption_words)


def search(parsed: ParsedQuery, top_k: int = 5, alpha: float = 0.6) -> list[ScoredResult]:
    scored: list[ScoredResult] = []

    for record in _CATALOG:
        symbolic, matched_fields = _symbolic_score(parsed, record)
        dense = _dense_score(parsed.raw_query, record)
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
