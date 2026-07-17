"""query_parsing.py tests: the orchestrator that picks between the real
LLM parser and the keyword-spotting fallback. No GEMINI_API_KEY is set
in the test environment, so parse_query() here should transparently
fall back to the keyword parser and match its behavior exactly.
"""

from app.schema import ParsedQuery
from app.services.query_parsing import parse_query


def test_falls_back_to_keyword_parser_when_not_configured():
    parsed = parse_query("A red tie and a white shirt in a formal setting")
    assert parsed.style == "formal"
    assert any(g.slot == "accessory" and g.color == "red" for g in parsed.garments)


def test_falls_back_when_llm_parser_raises_unexpectedly(monkeypatch):
    def boom(raw_query: str) -> ParsedQuery:
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr("app.services.query_parsing.parse_query_llm", boom)

    # Must not raise: an unexpected LLM-side failure should degrade to
    # the keyword parser, not take the whole request down with it.
    parsed = parse_query("A bright yellow raincoat")
    assert any(g.type == "raincoat" and g.color == "yellow" for g in parsed.garments)
