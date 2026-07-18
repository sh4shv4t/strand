"""image_similarity.py tests. The real-CLIP test proves the actual
wiring this module exists for: a query encoded through CLIP's text
tower finds the right image among embeddings stored by indexer.py,
CLIP's own text-image joint space doing the matching, not a mock.
"""

import sys

import pytest


def test_disabled_returns_empty_dict(monkeypatch):
    monkeypatch.setenv("STRAND_DISABLE_EMBEDDINGS", "1")

    from app.services.image_similarity import score

    assert score("anything") == {}


def test_missing_dependencies_returns_empty_dict_not_an_error(monkeypatch):
    import app.services.clip_model as clip_model_module
    import app.services.image_similarity as image_similarity_module

    monkeypatch.delenv("STRAND_DISABLE_EMBEDDINGS", raising=False)
    monkeypatch.setitem(sys.modules, "open_clip", None)
    monkeypatch.setattr(clip_model_module, "_model", None)
    monkeypatch.setattr(clip_model_module, "_preprocess", None)
    monkeypatch.setattr(clip_model_module, "_tokenizer", None)

    # Unlike DenseScorer's Chroma fallback, this is not "fall back to a
    # different scoring method", it is "contribute no image signal at
    # all", search()'s _blend_dense already treats a missing id as
    # caption-only, so an empty dict here degrades correctly on its own.
    assert image_similarity_module.score("anything") == {}


def test_real_similarity_prefers_the_matching_image(tmp_path, isolated_image_store, monkeypatch):
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    monkeypatch.delenv("STRAND_DISABLE_EMBEDDINGS", raising=False)
    from PIL import Image

    from app.services.image_similarity import score
    from app.services.indexer import index_image

    red_path = tmp_path / "red_square.jpg"
    blue_path = tmp_path / "blue_square.jpg"
    Image.new("RGB", (224, 224), color=(220, 20, 20)).save(red_path)
    Image.new("RGB", (224, 224), color=(20, 20, 220)).save(blue_path)

    index_image(str(red_path))
    index_image(str(blue_path))

    scores = score("a solid red square")

    assert scores.keys() == {"red_square", "blue_square"}
    assert scores["red_square"] > scores["blue_square"]


def test_cache_key_includes_collection_count_so_a_new_embedding_is_seen(
    tmp_path, isolated_image_store, monkeypatch
):
    """score() is cached on (raw_query, count) (see module docstring): a
    newly indexed image changes count, so a repeated identical query
    must see it immediately, with no explicit invalidation call needed,
    unlike vector_store.DenseScorer."""
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    monkeypatch.delenv("STRAND_DISABLE_EMBEDDINGS", raising=False)
    from PIL import Image

    import app.services.image_similarity as image_similarity_module
    from app.services.image_similarity import score
    from app.services.indexer import index_image

    # The module-level cache persists across tests in the same process;
    # cleared here so an entry another test left behind under the same
    # (query, count) key can't make this test pass for the wrong reason.
    image_similarity_module.clear_cache()

    red_path = tmp_path / "red_square.jpg"
    Image.new("RGB", (224, 224), color=(220, 20, 20)).save(red_path)
    index_image(str(red_path))

    before = score("a solid red square")
    assert before.keys() == {"red_square"}

    blue_path = tmp_path / "blue_square.jpg"
    Image.new("RGB", (224, 224), color=(20, 20, 220)).save(blue_path)
    index_image(str(blue_path))

    after = score("a solid red square")
    assert after.keys() == {"red_square", "blue_square"}
