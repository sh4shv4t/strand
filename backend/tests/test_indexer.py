"""indexer.py tests. The real-CLIP tests are skipped if open_clip is not
installed (it is a heavy optional dependency, see
scripts/requirements-eval.txt); the missing-dependencies test is
deterministic regardless of environment via monkeypatching sys.modules,
so it runs the same whether or not open_clip happens to be installed
locally.
"""

import sys

import pytest


@pytest.fixture()
def isolated_image_store(tmp_path, monkeypatch):
    """Real-indexing tests must not write into the actual persistent
    vector store used by production data (app/data/image_vector_index/).
    Redirects storage to a throwaway directory for the duration of the
    test, and resets the module-level singleton so a fresh client opens
    against it.
    """
    import app.services.image_vector_store as store_module

    monkeypatch.setattr(store_module, "PERSIST_DIR", tmp_path / "image_vector_index")
    monkeypatch.setattr(store_module, "_client", None)
    monkeypatch.setattr(store_module, "_collection", None)


def test_missing_dependencies_raises_clear_error(monkeypatch):
    import app.services.indexer as indexer_module

    monkeypatch.setitem(sys.modules, "open_clip", None)
    monkeypatch.setattr(indexer_module, "_model", None)
    monkeypatch.setattr(indexer_module, "_preprocess", None)

    with pytest.raises(indexer_module.IndexingDependenciesMissing, match="requirements-eval"):
        indexer_module.index_image("anything.jpg")


def test_missing_file_raises_file_not_found():
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    from app.services.indexer import index_image

    with pytest.raises(FileNotFoundError):
        index_image("this/path/does/not/exist.jpg")


def test_real_indexing_produces_an_embedding(tmp_path, isolated_image_store):
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    from PIL import Image

    from app.services.indexer import index_image

    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(120, 40, 200)).save(image_path)

    embedding = index_image(str(image_path))
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


def test_real_indexing_persists_to_the_vector_store(tmp_path, isolated_image_store):
    pytest.importorskip("open_clip")
    pytest.importorskip("torch")
    from PIL import Image

    from app.services.image_vector_store import get_image_collection
    from app.services.indexer import index_image

    image_path = tmp_path / "persisted.jpg"
    Image.new("RGB", (64, 64), color=(10, 200, 90)).save(image_path)

    index_image(str(image_path))

    collection = get_image_collection()
    stored = collection.get(ids=["persisted"])
    assert stored["ids"] == ["persisted"]
