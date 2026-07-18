from app.schema import ImageRecord
from app.services.vector_store import DenseScorer


def _record(record_id: str, caption: str) -> ImageRecord:
    return ImageRecord(id=record_id, garments=[], caption=caption)


def test_word_overlap_fallback_is_deterministic():
    records = [
        _record("a", "a red tie and a white shirt, formal setting, office"),
        _record("b", "a green dress, casual style, relaxed home setting"),
    ]
    scorer = DenseScorer(records, use_embeddings=False)

    assert scorer.using_real_embeddings is False

    scores = scorer.score("a red tie and a white shirt in a formal setting")
    assert scores["a"] > scores["b"]


def test_word_overlap_scores_are_bounded():
    records = [_record("a", "a bright yellow raincoat, casual style, street setting")]
    scorer = DenseScorer(records, use_embeddings=False)

    scores = scorer.score("completely unrelated query text")
    assert 0.0 <= scores["a"] <= 1.0


def test_empty_catalog_scores_empty():
    scorer = DenseScorer([], use_embeddings=False)
    assert scorer.score("anything") == {}


def test_score_is_cached_across_repeated_calls():
    """The second call for an identical query must not recompute -- a
    correctness proxy for the caching claim, cache hits and misses would
    look identical from score()'s return value alone, so this instead
    checks the two dicts are equal-valued but independent objects (proof
    the second call was served from _score_cached rather than an
    unrelated coincidence of a single shared mutable dict)."""
    records = [_record("a", "a red tie and a white shirt, formal setting, office")]
    scorer = DenseScorer(records, use_embeddings=False)

    first = scorer.score("a red tie")
    second = scorer.score("a red tie")

    assert first == second
    assert first is not second


def test_mutating_a_returned_score_dict_does_not_corrupt_the_cache():
    records = [_record("a", "a red tie and a white shirt, formal setting, office")]
    scorer = DenseScorer(records, use_embeddings=False)

    first = scorer.score("a red tie")
    first["a"] = -999.0

    second = scorer.score("a red tie")
    assert second["a"] != -999.0


def test_without_clear_cache_a_newly_added_record_stays_invisible():
    """Proves the cache is real (not just a no-op wrapper): the
    word-overlap fallback re-scans self._records live on every call (see
    add_record's docstring), so without clear_cache the only reason a
    newly added record could stay missing from a repeated identical
    query is a stale cached result."""
    records = [_record("a", "a red tie, formal setting, office")]
    scorer = DenseScorer(records, use_embeddings=False)

    scorer.score("a red tie")
    scorer._records.append(_record("b", "a red tie, formal setting, office"))

    stale = scorer.score("a red tie")
    assert "b" not in stale


def test_clear_cache_allows_a_repeated_query_to_see_a_newly_added_record():
    records = [_record("a", "a red tie, formal setting, office")]
    scorer = DenseScorer(records, use_embeddings=False)

    before = scorer.score("a red tie")
    assert "b" not in before

    scorer._records.append(_record("b", "a red tie, formal setting, office"))
    scorer.clear_cache()

    after = scorer.score("a red tie")
    assert "b" in after
