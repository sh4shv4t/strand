"""Weighted-hybrid retrieval scoring.

Implements `score = alpha * symbolic_match + (1 - alpha) * dense_similarity`
from Working_notes.md Section 3. Symbolic matching lives here; dense
similarity is delegated to vector_store.DenseScorer, and catalog loading to
catalog.load_catalog, this module only blends the two.

"dense_similarity" is itself a blend of two independent real signals when
both exist for a record: caption-text similarity (DenseScorer, always
available) and real CLIP image-pixel similarity (image_similarity,
available only for records with a stored embedding, the 40 real
Fashionpedia photos, not the 12 hand-written mock records). This is what
actually wires Part A's real image embeddings into ranking, previously
index_image() persisted them and nothing ever read them back. See
_blend_dense and Working_notes.md Section 14 for why a plain mean, not a
new tunable weight.

Scales alpha by parsed.confidence (see _effective_alpha): a query the
parser only partially recognized should trust its symbolic signal
proportionally less, not get the same fixed weight as a fully-recognized
query. A query with no structured signal at all has confidence pinned to
0 effectively, since symbolic_score is 0.0 for every record regardless of
alpha in that case, so dense similarity is reported at full strength
rather than discounted for a symbolic signal that doesn't exist.

search() scores every record in the catalog directly, no symbolic
pre-filter. One was built and benchmarked (an in-process inverted index,
and before that a Chroma metadata `where` filter) and both measured
*slower* than this plain loop at every scale up to 1,000,000 records,
including at 4.7% candidate-set selectivity. The reason is structural,
not a tuning gap: this is a hybrid ranking, so a record with symbolic
score 0 can still win on dense score alone, meaning dense similarity
must be computed for every record regardless of any symbolic filter.
Pre-filtering can therefore only ever skip the ~2 microsecond
_symbolic_score call itself, replacing it with a same-cost set
membership check plus the fixed cost of building the candidate set, a
trade with no winning scale. See Working_notes.md Section 13 for the
full investigation and measured numbers.
"""

from app.schema import Garment, ImageRecord, ParsedQuery, ScoredResult
from app.services import image_similarity
from app.services.catalog import get_catalog
from app.services.garment_vocabulary import canonical_type
from app.services.vector_store import DenseScorer

_CATALOG: list[ImageRecord] = get_catalog()
_DENSE_SCORER = DenseScorer(_CATALOG)


def get_dense_scorer() -> DenseScorer:
    return _DENSE_SCORER


def _garment_matches(query_garment: Garment, record_garments: list[Garment]) -> bool:
    for rg in record_garments:
        if rg.slot != query_garment.slot:
            continue
        if query_garment.type and canonical_type(query_garment.type) != canonical_type(rg.type):
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


def _effective_alpha(parsed: ParsedQuery, alpha: float) -> float:
    # No structured signal at all: symbolic_score is 0.0 for every record
    # regardless of alpha, so this is a genuinely zero-shot query. Using
    # alpha=0 here reports the dense score at full strength instead of
    # discounting it for a symbolic signal that doesn't exist -- ranking
    # is unaffected either way (every record scaled the same amount), but
    # a perfect dense hit would otherwise misleadingly show as 40% instead
    # of 100%.
    if not _has_structured_signal(parsed):
        return 0.0

    # Some signal, but not necessarily all of it: parsed.confidence (0.4 to
    # 1.0 from the keyword parser depending on how much of the query it
    # recognized, always 1.0 from a successful real LLM parse) scales alpha
    # proportionally, so a query the parser only half-understood trusts
    # its symbolic match less and leans further on dense similarity,
    # rather than getting the same fixed weight as a fully-recognized one.
    return alpha * parsed.confidence


def _blend_dense(caption_sim: float, image_sim: float | None) -> float:
    # A plain mean, not a new tunable weight: this project already has one
    # unvalidated hyperparameter (alpha, see Working_notes.md Section 8),
    # adding a second before either is validated against real data would
    # compound the problem rather than fix anything. Records with no
    # stored image embedding (the 12 hand-written mock records) fall back
    # to caption similarity alone, exactly the pre-existing behavior.
    if image_sim is None:
        return caption_sim
    return (caption_sim + image_sim) / 2


def search(parsed: ParsedQuery, top_k: int = 5, alpha: float = 0.6) -> list[ScoredResult]:
    effective_alpha = _effective_alpha(parsed, alpha)

    caption_scores = _DENSE_SCORER.score(parsed.raw_query)
    image_scores = image_similarity.score(parsed.raw_query)

    scored: list[ScoredResult] = []

    for record in _CATALOG:
        symbolic, matched_fields = _symbolic_score(parsed, record)
        dense = _blend_dense(caption_scores.get(record.id, 0.0), image_scores.get(record.id))
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
