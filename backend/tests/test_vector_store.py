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
