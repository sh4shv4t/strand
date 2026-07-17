"""llm_query_parser.py tests. See test_gemini_client.py for why these
don't make a real API call."""

import pytest

from app.services.gemini_client import GeminiNotConfigured
from app.services.llm_query_parser import parse_query_llm


def test_raises_when_not_configured(monkeypatch):
    import app.services.gemini_client as gemini_client_module

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_client_module, "_client", None)

    with pytest.raises(GeminiNotConfigured):
        parse_query_llm("a red tie and a white shirt in a formal setting")


@pytest.mark.skipif(
    "not __import__('os').environ.get('GEMINI_API_KEY')",
    reason="needs a real GEMINI_API_KEY to actually call the model",
)
def test_real_call_parses_the_compositional_query():
    parsed = parse_query_llm("A red tie and a white shirt in a formal setting")
    assert parsed.style == "formal"
    assert any(g.slot == "accessory" and g.color == "red" for g in parsed.garments)
    assert any(g.slot == "upper" and g.color == "white" for g in parsed.garments)
