import os

# Must be set before app.services.retriever is first imported (it builds the
# dense scorer at module import time) -- keeps tests offline/deterministic.
os.environ.setdefault("STRAND_DISABLE_EMBEDDINGS", "1")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def isolated_image_store(tmp_path, monkeypatch):
    """Real-indexing tests must not write into the actual persistent
    vector store used by production data (app/data/image_vector_index/).
    Redirects storage to a throwaway directory for the duration of the
    test, and resets the module-level singleton so a fresh client opens
    against it. Shared by test_indexer.py and test_image_similarity.py.
    """
    import app.services.image_vector_store as store_module

    monkeypatch.setattr(store_module, "PERSIST_DIR", tmp_path / "image_vector_index")
    monkeypatch.setattr(store_module, "_client", None)
    monkeypatch.setattr(store_module, "_collection", None)
