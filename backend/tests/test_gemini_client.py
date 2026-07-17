"""gemini_client.py tests. No GEMINI_API_KEY is set in the test
environment (see conftest.py / .github/workflows/ci.yml), so these
exercise the deterministic "not configured" path. A real call would need
an actual key and would cost real money per test run, exactly the kind
of thing kept out of the automated suite elsewhere in this repo (see
vector_store.py's STRAND_DISABLE_EMBEDDINGS, or indexer.py's heavy
optional deps).
"""

import pytest

from app.services.gemini_client import GeminiNotConfigured, get_client


def test_get_client_raises_when_not_configured(monkeypatch):
    import app.services.gemini_client as gemini_client_module

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_client_module, "_client", None)

    with pytest.raises(GeminiNotConfigured):
        get_client()
