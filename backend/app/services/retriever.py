"""Weighted-hybrid retrieval scoring.

Implements `score = alpha * symbolic_match + (1 - alpha) * dense_similarity`
from Working_notes.md Section 3. Symbolic matching lives here; dense
similarity is delegated to vector_store.DenseScorer, and catalog loading to
catalog.load_catalog, this module only blends the two.

Falls back to alpha=0 for queries with no structured signal at all (see
_has_structured_signal): with no garments, scene, or style recognized,
symbolic_score is 0.0 for every record, so the query is genuinely zero-shot
and dense similarity should be reported at full strength rather than
discounted for a symbolic signal that doesn't exist.
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


def _has_structured_signal(parsed: ParsedQuery) -> bool:
    return bool(parsed.garments or parsed.scene or parsed.style)


def search(parsed: ParsedQuery, top_k: int = 5, alpha: float = 0.6) -> list[ScoredResult]:
    # When the parser recognizes nothing (a query outside its keyword
    # vocabulary), symbolic_score is 0.0 for every record regardless of
    # alpha, so this is a genuinely zero-shot query: dense similarity is
    # the only real signal available. Discounting it by (1 - alpha) here
    # doesn't change the ranking (every record is scaled the same amount),
    # but it does understate how good the match actually is -- a perfect
    # dense hit would otherwise show as a 40% match instead of 100%. Using
    # alpha=0 in this case reports the dense score at full strength.
    effective_alpha = alpha if _has_structured_signal(parsed) else 0.0

    dense_scores = _DENSE_SCORER.score(parsed.raw_query)
    scored: list[ScoredResult] = []

    for record in _CATALOG:
        symbolic, matched_fields = _symbolic_score(parsed, record)
        dense = dense_scores.get(record.id, 0.0)
        score = effective_alpha * symbolic + (1 - effective_alpha) * dense

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
