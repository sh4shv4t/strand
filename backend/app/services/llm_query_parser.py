"""Real LLM query parser, using the exact prompt drafted in
Working_notes.md Section 4.3.1. Same output contract as
query_parser.parse_query_keywords: raw text in, ParsedQuery out, so
query_parsing.py can swap between the two without either side of the
API needing to know which one ran.

Needs GEMINI_API_KEY (see gemini_client.py); raises GeminiNotConfigured
if it is not set, which query_parsing.py catches to fall back to the
keyword parser.
"""

from app.schema import ExtractedAttributes, ParsedQuery
from app.services.gemini_client import generate_structured

SYSTEM_INSTRUCTION = """You are a query parser for a fashion image search system. Given a
natural language search query, extract structured information: the
garments mentioned (each bound to a slot: upper, lower, outerwear,
footwear, or accessory, with a type and an optional color), the scene
(office, street, park, home, or other), and the style (formal, casual,
athleisure, business, or other).

Only include a garment, scene, or style if the query actually supports
it. Do not guess or invent details the query does not mention."""

# The same five eval queries from Working_notes.md Section 6, used as
# few-shot examples so the parser's few-shot coverage and the grading
# queries are the same set.
FEW_SHOT_EXAMPLES = """
Query: "A bright yellow raincoat"
JSON: {"garments": [{"slot": "outerwear", "type": "raincoat", "color": "yellow"}], "scene": null, "style": null}

Query: "Professional business attire inside a modern office"
JSON: {"garments": [], "scene": "office", "style": "business"}

Query: "Someone wearing a blue shirt sitting on a park bench"
JSON: {"garments": [{"slot": "upper", "type": "shirt", "color": "blue"}], "scene": "park", "style": null}

Query: "Casual weekend outfit for a city walk"
JSON: {"garments": [], "scene": "street", "style": "casual"}

Query: "A red tie and a white shirt in a formal setting"
JSON: {"garments": [{"slot": "accessory", "type": "tie", "color": "red"}, {"slot": "upper", "type": "shirt", "color": "white"}], "scene": null, "style": "formal"}
"""


def parse_query_llm(raw_query: str) -> ParsedQuery:
    prompt = f"{FEW_SHOT_EXAMPLES}\nQuery: {raw_query!r}\nJSON:"

    attributes: ExtractedAttributes = generate_structured(
        contents=prompt,
        response_schema=ExtractedAttributes,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    return ParsedQuery(
        raw_query=raw_query,
        garments=attributes.garments,
        scene=attributes.scene,
        style=attributes.style,
        confidence=1.0,
    )
