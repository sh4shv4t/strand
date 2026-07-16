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
