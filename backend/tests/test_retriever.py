"""Retriever tests run with STRAND_DISABLE_EMBEDDINGS=1 (set in conftest.py),
so the dense half is the deterministic word-overlap fallback -- these tests
exercise symbolic scoring and score blending against the real catalog."""

import pytest

from app.schema import ImageRecord, ParsedQuery, ScoredResult
from app.services.retriever import (
    _blend_dense,
    _canonical_phrase,
    _reciprocal_rank_fusion,
    get_catalog,
    register_record,
    search,
)


def _find(results, record_id: str):
    return next(r for r in results if r.record.id == record_id)


def test_compositional_decoy_ranks_correctly():
    """The whole point of the schema-based approach: a color-swapped decoy
    (img_006) must not outrank the true match (img_005) even though their
    captions are lexically almost identical."""
    parsed = ParsedQuery(
        raw_query="a red tie and a white shirt in a formal setting",
        garments=[
            {"slot": "upper", "type": "shirt", "color": "white"},
            {"slot": "accessory", "type": "tie", "color": "red"},
        ],
        style="formal",
    )
    results = search(parsed, top_k=len(get_catalog()))

    assert results[0].record.id == "img_005"
    true_match = _find(results, "img_005")
    decoy = _find(results, "img_006")
    assert true_match.score > decoy.score
    assert true_match.symbolic_score == 1.0
    assert decoy.symbolic_score < 1.0


def test_search_respects_top_k():
    parsed = ParsedQuery(raw_query="a bright yellow raincoat")
    results = search(parsed, top_k=3)
    assert len(results) == 3


def test_no_structured_signal_gives_zero_symbolic_score():
    parsed = ParsedQuery(raw_query="some free text with no schema signal")
    results = search(parsed, top_k=len(get_catalog()))
    assert all(r.symbolic_score == 0.0 for r in results)


def test_no_structured_signal_uses_dense_score_at_full_strength():
    """A query the parser recognizes nothing in is a genuinely zero-shot
    query. The final score should equal the dense score exactly (alpha=0
    in that case), not the dense score discounted by a fixed alpha that
    was only ever meant to weight a symbolic signal that doesn't exist
    here. Without this, a perfect dense match would misleadingly show as
    a 40% match instead of 100%."""
    parsed = ParsedQuery(raw_query="some free text with no schema signal")
    results = search(parsed, top_k=len(get_catalog()), alpha=0.6)
    for r in results:
        assert r.score == r.dense_score


def test_lower_confidence_scales_alpha_down():
    """A parse the parser was less sure about should trust its symbolic
    signal proportionally less, not get the same fixed alpha weight as a
    fully-confident parse. Same garments/scene/style, only confidence
    differs, score should follow score = (alpha*confidence)*symbolic +
    (1 - alpha*confidence)*dense exactly."""
    base_kwargs = dict(
        raw_query="a red tie and a white shirt in a formal setting",
        garments=[
            {"slot": "upper", "type": "shirt", "color": "white"},
            {"slot": "accessory", "type": "tie", "color": "red"},
        ],
        style="formal",
    )
    confident = ParsedQuery(**base_kwargs, confidence=1.0)
    unsure = ParsedQuery(**base_kwargs, confidence=0.5)

    confident_result = _find(search(confident, top_k=len(get_catalog())), "img_005")
    unsure_result = _find(search(unsure, top_k=len(get_catalog())), "img_005")

    # Same symbolic/dense components either way, only the blend differs.
    assert confident_result.symbolic_score == unsure_result.symbolic_score
    assert confident_result.dense_score == unsure_result.dense_score

    alpha = 0.6
    symbolic, dense = confident_result.symbolic_score, confident_result.dense_score
    expected_confident = round(alpha * 1.0 * symbolic + (1 - alpha * 1.0) * dense, 4)
    expected_unsure = round(alpha * 0.5 * symbolic + (1 - alpha * 0.5) * dense, 4)

    assert confident_result.score == expected_confident
    assert unsure_result.score == expected_unsure
    assert unsure_result.score != confident_result.score


def test_garment_synonym_matches_a_differently_worded_type():
    """Before garment_vocabulary.py's canonicalization, this failed
    silently: the keyword parser's plural "shoes" never matched any real
    catalog record, all of which are typed "shoe" (Fashionpedia's
    singular ground-truth category). fp_6410 is a real catalog record
    with a "shoe" garment."""
    parsed = ParsedQuery(raw_query="shoes", garments=[{"slot": "footwear", "type": "shoes"}])
    results = search(parsed, top_k=len(get_catalog()))
    match = _find(results, "fp_6410")
    assert "shoes" in match.matched_fields


def test_blend_dense_averages_when_image_signal_exists():
    assert _blend_dense(0.4, 0.8) == pytest.approx(0.6)


def test_blend_dense_falls_back_to_caption_only_without_an_image_embedding():
    """The 12 hand-written mock records have no real photo, so no stored
    CLIP embedding, image_similarity.score() simply won't have an entry
    for them, this is that case."""
    assert _blend_dense(0.4, None) == 0.4


def test_register_record_makes_a_new_record_visible_to_a_repeated_query():
    """DenseScorer.score() is cached (Working_notes.md Section 18): a
    query run once before a record is registered must still find it on
    a repeat, proving register_record's clear_cache() call actually
    drops the stale entry rather than leaving it to expire on its own."""
    parsed = ParsedQuery(raw_query="a distinctive unregistered garment query")

    before_ids = {r.record.id for r in search(parsed, top_k=len(get_catalog()))}
    assert "new_test_record" not in before_ids

    register_record(
        ImageRecord(
            id="new_test_record",
            garments=[{"slot": "upper", "type": "shirt", "color": "white"}],
            caption="a distinctive unregistered garment query",
        )
    )

    after_ids = {r.record.id for r in search(parsed, top_k=len(get_catalog()))}
    assert "new_test_record" in after_ids


def test_canonical_phrase_reuses_matched_fields_label_format():
    parsed = ParsedQuery(
        raw_query="irrelevant",
        garments=[
            {"slot": "upper", "type": "shirt", "color": "white"},
            {"slot": "accessory", "type": "tie", "color": "red"},
        ],
        scene="office",
        style="formal",
    )
    assert _canonical_phrase(parsed) == "white shirt, red tie, office, formal"


def test_canonical_phrase_is_none_without_any_structured_signal():
    parsed = ParsedQuery(raw_query="some free text with no schema signal")
    assert _canonical_phrase(parsed) is None


def _fusion_candidate(record_id: str, score: float) -> ScoredResult:
    return ScoredResult(
        record=ImageRecord(id=record_id, garments=[], caption=""),
        score=score,
        symbolic_score=0.0,
        dense_score=score,
        matched_fields=[],
    )


def test_reciprocal_rank_fusion_promotes_a_record_that_ranks_well_in_both_lists():
    """"b" has the single highest raw score in `primary`, but ranks last
    in `secondary`. "a" ranks 2nd/1st across the two lists, better on
    average, RRF (rank-based, not score-based) must place it ahead of
    "b" despite "b" winning on primary's score alone."""
    primary = [_fusion_candidate("a", 0.5), _fusion_candidate("b", 0.9), _fusion_candidate("c", 0.1)]
    secondary = [_fusion_candidate("a", 0.9), _fusion_candidate("c", 0.5), _fusion_candidate("b", 0.1)]

    fused = _reciprocal_rank_fusion(primary, secondary)
    fused_ids = [r.record.id for r in fused]

    assert fused_ids.index("a") < fused_ids.index("b")


def test_reciprocal_rank_fusion_returns_primarys_own_result_objects():
    """The score/symbolic_score/dense_score/matched_fields a caller sees
    must always come from `primary` (the raw-query pass), never
    `secondary`, RRF only reorders, it never substitutes a different
    scoring pass's reported numbers."""
    primary = [_fusion_candidate("a", 0.5), _fusion_candidate("b", 0.9)]
    secondary = [_fusion_candidate("a", 0.9), _fusion_candidate("b", 0.1)]

    fused = _reciprocal_rank_fusion(primary, secondary)

    by_id = {r.record.id: r for r in fused}
    assert by_id["a"].score == 0.5
    assert by_id["b"].score == 0.9


def test_rrf_only_changes_order_not_the_reported_score_fields(monkeypatch):
    """search() with RRF enabled must report the exact same
    score/symbolic_score/dense_score/matched_fields per record as the
    raw-query-only pass, RRF is only ever allowed to reorder the list."""
    parsed = ParsedQuery(
        raw_query="some vague phrasing that barely resembles anything",
        garments=[{"slot": "upper", "type": "shirt", "color": "white"}],
        style="business",
    )

    with_rrf = {r.record.id: r for r in search(parsed, top_k=len(get_catalog()))}

    import app.services.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "_canonical_phrase", lambda p: None)
    without_rrf = {r.record.id: r for r in search(parsed, top_k=len(get_catalog()))}

    assert with_rrf.keys() == without_rrf.keys()
    for record_id, result in with_rrf.items():
        baseline = without_rrf[record_id]
        assert result.score == baseline.score
        assert result.symbolic_score == baseline.symbolic_score
        assert result.dense_score == baseline.dense_score
        assert result.matched_fields == baseline.matched_fields


def test_scene_and_style_match_contribute_to_score():
    parsed = ParsedQuery(raw_query="office style query", scene="office", style="business")
    results = search(parsed, top_k=len(get_catalog()))
    matched = [r for r in results if r.record.scene == "office" and r.record.style == "business"]
    assert matched, "expected at least one office/business record in the catalog"
    for r in matched:
        assert "scene:office" in r.matched_fields
        assert "style:business" in r.matched_fields
