"""Retriever tests run with STRAND_DISABLE_EMBEDDINGS=1 (set in conftest.py),
so the dense half is the deterministic word-overlap fallback -- these tests
exercise symbolic scoring and score blending against the real catalog."""

from app.schema import ParsedQuery
from app.services.retriever import get_catalog, search


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


def test_scene_and_style_match_contribute_to_score():
    parsed = ParsedQuery(raw_query="office style query", scene="office", style="business")
    results = search(parsed, top_k=len(get_catalog()))
    matched = [r for r in results if r.record.scene == "office" and r.record.style == "business"]
    assert matched, "expected at least one office/business record in the catalog"
    for r in matched:
        assert "scene:office" in r.matched_fields
        assert "style:business" in r.matched_fields
