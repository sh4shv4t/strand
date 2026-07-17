"""Exercises the keyword-spotting fallback parser against the 5 eval
queries from Working_notes.md Section 6 -- the same set drafted as
few-shot examples for the real LLM parser in Section 4.3.1. See
test_query_parsing.py for the orchestrator that picks between this and
the real parser, and test_llm_query_parser.py for the real one."""

from app.services.query_parser import parse_query_keywords as parse_query


def _has_garment(parsed, slot: str, garment_type: str, color: str | None = None) -> bool:
    return any(
        g.slot == slot and g.type == garment_type and (color is None or g.color == color)
        for g in parsed.garments
    )


def test_bright_yellow_raincoat():
    parsed = parse_query("A bright yellow raincoat")
    assert _has_garment(parsed, "outerwear", "raincoat", "yellow")
    assert parsed.scene is None
    assert parsed.style is None


def test_professional_business_attire_in_office():
    parsed = parse_query("Professional business attire inside a modern office")
    assert parsed.scene == "office"
    assert parsed.style == "business"


def test_blue_shirt_on_park_bench():
    parsed = parse_query("Someone wearing a blue shirt sitting on a park bench")
    assert _has_garment(parsed, "upper", "shirt", "blue")
    assert parsed.scene == "park"


def test_casual_weekend_city_walk():
    parsed = parse_query("Casual weekend outfit for a city walk")
    assert parsed.scene == "street"
    assert parsed.style == "casual"
    assert parsed.garments == []


def test_red_tie_white_shirt_formal():
    parsed = parse_query("A red tie and a white shirt in a formal setting")
    assert _has_garment(parsed, "accessory", "tie", "red")
    assert _has_garment(parsed, "upper", "shirt", "white")
    assert parsed.style == "formal"


def test_unmatched_query_has_low_confidence():
    parsed = parse_query("something with no keywords in it at all")
    assert parsed.garments == []
    assert parsed.scene is None
    assert parsed.style is None
    assert parsed.confidence < 1.0
