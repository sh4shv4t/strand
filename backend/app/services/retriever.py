"""Weighted-hybrid retrieval scoring.

Implements `score = alpha * symbolic_match + (1 - alpha) * dense_similarity`
from Working_notes.md Section 3. Symbolic matching lives here; dense
similarity is delegated to vector_store.DenseScorer, and catalog loading to
catalog.load_catalog, this module only blends the two.

"dense_similarity" is itself a blend of up to three independent real
signals when they exist for a record: caption-text similarity
(DenseScorer, always available), real CLIP image-pixel similarity
(image_similarity, available only for records with a stored embedding,
the real Fashionpedia photos, not the 12 hand-written mock records), and
a soft color-similarity signal for garments whose color came from RGB
detection rather than a confident source (see _color_signal and
Working_notes.md Section 17). This is what actually wires Part A's real
image embeddings into ranking, previously index_image() persisted them
and nothing ever read them back. See _blend_dense and Working_notes.md
Section 14 for why a plain mean, not a new tunable weight.

Detected colors get a *soft* signal instead of hard symbolic matching
specifically because they can be wrong in a way hand-authored or
VLM-confirmed colors are much less likely to be (Working_notes.md
Section 16's own measured example): a wrong confident color would have
been a labeling bug, a wrong detected color is an expected, nonzero
rate of a heuristic. Hard-gating on it would silently turn "unknown
color" (never excludes) into "confidently wrong" (wrongly excludes) for
every detection error, worse than not detecting color at all.

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

When the parser found structured fields, search() also fuses in a
second ranking via Reciprocal Rank Fusion (see _reciprocal_rank_fusion
and Working_notes.md Section 19): the same scoring formula run again
with the raw query text swapped for a canonical phrase synthesized
straight from parsed.garments/scene/style (_canonical_phrase). The raw
query is whatever free-text phrasing the user typed, "something in a
reddish jacket for the office maybe", the canonical phrase is the
parser's own clean read of that, "red jacket, business". Dense
similarity (CLIP/caption embeddings) is sensitive to exactly this kind
of surface phrasing in a way symbolic matching never is, so a query
that embeds poorly in its original wording can still rank well once its
structured meaning is re-rendered as cleaner text. RRF combines the two
resulting rank positions (not scores, ranks: `1/(k+rank)` per list, the
standard 60 default for k), so a record that ranks well under either
phrasing floats up without either list dominating outright. This only
ever changes final ordering, the returned ScoredResult.score,
symbolic_score, dense_score, and matched_fields fields all still come
from the original raw-query scoring pass untouched. Skipped entirely
when there's no structured signal to build a canonical phrase from (a
genuinely free-text query has nothing to fuse against, see
_has_structured_signal), or when the canonical phrase is identical to
the raw query (nothing to gain from ranking the same text against
itself twice).
"""

from app.schema import Garment, ImageRecord, ParsedQuery, ScoredResult
from app.services import image_similarity
from app.services.catalog import get_catalog
from app.services.color_detection import color_similarity
from app.services.garment_vocabulary import canonical_type
from app.services.vector_store import DenseScorer

_CATALOG: list[ImageRecord] = get_catalog()
_DENSE_SCORER = DenseScorer(_CATALOG)


def get_dense_scorer() -> DenseScorer:
    return _DENSE_SCORER


def register_record(record: ImageRecord) -> None:
    """Adds a newly indexed image to the live catalog and its caption
    index so it is immediately searchable via search(). This is the
    cold-start fix: routers/index.py's index_image() call already
    persisted a real CLIP embedding for a new image (Part A), but
    nothing added the image to the catalog itself, so it could never
    actually be returned by a query. _CATALOG is the same list object
    get_catalog() returns (not a copy), so appending here is visible to
    every other module holding that reference, no separate propagation
    needed.

    image_similarity needs no equivalent registration call: unlike
    DenseScorer's caption collection, it re-queries image_vector_store's
    persistent collection directly on every call, so a newly stored
    embedding is already visible there the moment index_image() persists
    it, before this function ever runs.

    DenseScorer.score() is cached (Working_notes.md Section 18), so a
    query cached before this record existed would otherwise keep
    returning a result missing it, clear_cache() drops that stale state.
    image_similarity's cache doesn't need the same call, its cache key
    already includes the collection's size, so it invalidates itself the
    moment a new embedding changes that count.
    """
    _CATALOG.append(record)
    _DENSE_SCORER.add_record(record)
    _DENSE_SCORER.clear_cache()


def _garment_matches(
    query_garment: Garment, record_garments: list[Garment], detected_color_slots: frozenset[str] = frozenset()
) -> bool:
    for rg in record_garments:
        if rg.slot != query_garment.slot:
            continue
        if query_garment.type and canonical_type(query_garment.type) != canonical_type(rg.type):
            continue
        # A confident color (hand-authored, or eventually VLM-confirmed)
        # still hard-gates a mismatch. A detected color does not, see the
        # module docstring and Working_notes.md Section 17, it's handled
        # as a soft signal in _color_signal/_blend_dense instead.
        color_is_confident = rg.slot not in detected_color_slots
        if color_is_confident and query_garment.color and rg.color and query_garment.color != rg.color:
            continue
        return True
    return False


def _color_signal(parsed: ParsedQuery, record: ImageRecord) -> float | None:
    """Soft color-match score, the counterpart to _garment_matches' hard
    gate for confident colors: for a query garment with a color, compare
    it against any same-slot record garment whose color came from RGB
    detection, via graded RGB similarity rather than exact-name equality.
    Returns None (not 0.0) when there's nothing to compare, no query
    color, or no detected-color garment in a matching slot, so
    _blend_dense can leave this signal out of the average instead of
    diluting it with an irrelevant zero.
    """
    if not record.detected_color_slots:
        return None

    detected_slots = frozenset(record.detected_color_slots)
    best: float | None = None
    for g in parsed.garments:
        if not g.color or g.slot not in detected_slots:
            continue
        for rg in record.garments:
            if rg.slot != g.slot or not rg.color:
                continue
            similarity = color_similarity(g.color, rg.color)
            if best is None or similarity > best:
                best = similarity
    return best


def _symbolic_score(parsed: ParsedQuery, record: ImageRecord) -> tuple[float, list[str]]:
    total_fields = len(parsed.garments) + (1 if parsed.scene else 0) + (1 if parsed.style else 0)
    if total_fields == 0:
        return 0.0, []

    detected_slots = frozenset(record.detected_color_slots)
    matched_fields: list[str] = []

    for g in parsed.garments:
        if _garment_matches(g, record.garments, detected_slots):
            label = f"{g.color + ' ' if g.color else ''}{g.type}".strip()
            matched_fields.append(label)

    if parsed.scene and parsed.scene == record.scene:
        matched_fields.append(f"scene:{parsed.scene}")

    if parsed.style and parsed.style == record.style:
        matched_fields.append(f"style:{parsed.style}")

    matched_count = (
        sum(1 for g in parsed.garments if _garment_matches(g, record.garments, detected_slots))
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


def _blend_dense(caption_sim: float, image_sim: float | None, color_sim: float | None = None) -> float:
    # A plain mean over whichever signals actually apply, not a new
    # tunable weight: this project already has one unvalidated
    # hyperparameter (alpha, see Working_notes.md Section 8), adding
    # another before either is validated against real data would compound
    # the problem rather than fix anything. A record with no stored image
    # embedding (the 12 hand-written mock records), or a query/record with
    # nothing to compare on color, simply drops that term from the
    # average rather than diluting it with an irrelevant value.
    parts = [caption_sim]
    if image_sim is not None:
        parts.append(image_sim)
    if color_sim is not None:
        parts.append(color_sim)
    return sum(parts) / len(parts)


def _canonical_phrase(parsed: ParsedQuery) -> str | None:
    """The parser's own clean restatement of what it understood, reusing
    the exact "color type" label format _symbolic_score already builds
    for matched_fields (not the pull_fashionpedia_sample.py caption
    format, which also parenthesizes the slot, that extra token only
    dilutes word-overlap/embedding similarity against the free-text
    style of the 12 hand-written mock captions for no benefit). Returns
    None when there is no structured signal at all to build one from.
    """
    if not _has_structured_signal(parsed):
        return None

    parts = [f"{(g.color + ' ') if g.color else ''}{g.type}".strip() for g in parsed.garments]
    if parsed.scene:
        parts.append(parsed.scene)
    if parsed.style:
        parts.append(parsed.style)
    return ", ".join(parts)


_RRF_K = 60


def _reciprocal_rank_fusion(primary: list[ScoredResult], secondary: list[ScoredResult]) -> list[ScoredResult]:
    """Fuses two independently-ranked passes over the same catalog by
    rank position, not score magnitude, the standard RRF formula
    `1/(k+rank)` per list, k=60. Returns primary's own ScoredResult
    objects re-ordered, never secondary's, so the score/symbolic_score/
    dense_score/matched_fields a caller sees always reflect the raw
    query, RRF only ever moves records around in the final list.
    """
    primary_rank = {r.record.id: i for i, r in enumerate(sorted(primary, key=lambda r: r.score, reverse=True))}
    secondary_rank = {r.record.id: i for i, r in enumerate(sorted(secondary, key=lambda r: r.score, reverse=True))}

    def rrf_score(result: ScoredResult) -> float:
        return 1.0 / (_RRF_K + primary_rank[result.record.id] + 1) + 1.0 / (
            _RRF_K + secondary_rank[result.record.id] + 1
        )

    return sorted(primary, key=rrf_score, reverse=True)


def search(parsed: ParsedQuery, top_k: int = 5, alpha: float = 0.6) -> list[ScoredResult]:
    effective_alpha = _effective_alpha(parsed, alpha)

    # symbolic_score, matched_fields, and color_sim depend only on
    # parsed.garments/scene/style against each record, never on which
    # query text is being scored densely, computed once and reused for
    # both the raw-query and canonical-phrase passes below rather than
    # redoing identical work twice per record.
    precomputed = []
    for record in _CATALOG:
        symbolic, matched_fields = _symbolic_score(parsed, record)
        color_sim = _color_signal(parsed, record)
        precomputed.append((record, symbolic, matched_fields, color_sim))

    def score_against(query_text: str) -> list[ScoredResult]:
        caption_scores = _DENSE_SCORER.score(query_text)
        image_scores = image_similarity.score(query_text)
        scored: list[ScoredResult] = []
        for record, symbolic, matched_fields, color_sim in precomputed:
            dense = _blend_dense(caption_scores.get(record.id, 0.0), image_scores.get(record.id), color_sim)
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
        return scored

    primary = score_against(parsed.raw_query)

    canonical_phrase = _canonical_phrase(parsed)
    if canonical_phrase is None or canonical_phrase == parsed.raw_query:
        primary.sort(key=lambda r: r.score, reverse=True)
        return primary[:top_k]

    secondary = score_against(canonical_phrase)
    fused = _reciprocal_rank_fusion(primary, secondary)
    return fused[:top_k]
