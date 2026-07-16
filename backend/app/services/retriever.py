"""Weighted-hybrid retrieval scoring.

Implements `score = alpha * symbolic_match + (1 - alpha) * dense_similarity`
from Working_notes.md Section 3. Symbolic matching lives here; dense
similarity is delegated to vector_store.DenseScorer, and catalog loading to
catalog.load_catalog -- this module only blends the two.
"""

from app.schema import Garment, ImageRecord, ParsedQuery, ScoredResult
from app.services.catalog import get_catalog
from app.services.vector_store import DenseScorer

_CATALOG: list[ImageRecord] = get_catalog()
_DENSE_SCORER = DenseScorer(_CATALOG)


def get_dense_scorer() -> DenseScorer:
    return _DENSE_SCORER


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
    dense_scores = _DENSE_SCORER.score(parsed.raw_query)
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
