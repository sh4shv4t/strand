"""Keyword-spotting query parser: the fallback path.

query_parsing.py tries the real LLM parser (llm_query_parser.py) first
and falls back to parse_query_keywords() here whenever Gemini is not
configured or a call fails, so the app always answers a query rather
than erroring out. Same input/output contract either way: raw NL query
in, ParsedQuery out.

Approach: keyword spotting for colors/garment-types/scenes/styles, with
a garment's color taken from the color word immediately preceding it in
the query. This is intentionally simple and will misparse anything the
keyword lists don't cover -- that's expected of a fallback, not a bug.
"""

import re

from app.schema import Garment, ParsedQuery
from app.services.colors import COLOR_HEX

COLORS = list(COLOR_HEX)

# garment type -> slot it occupies
GARMENT_SLOTS = {
    "raincoat": "outerwear",
    "blazer": "outerwear",
    "jacket": "outerwear",
    "coat": "outerwear",
    "shirt": "upper",
    "blouse": "upper",
    "t-shirt": "upper",
    "tshirt": "upper",
    "tank top": "upper",
    "dress": "upper",
    "hoodie": "upper",
    "top": "upper",
    "trousers": "lower",
    "pants": "lower",
    "jeans": "lower",
    "shorts": "lower",
    "skirt": "lower",
    "leggings": "lower",
    "joggers": "lower",
    "sneakers": "footwear",
    "heels": "footwear",
    "boots": "footwear",
    "sandals": "footwear",
    "shoes": "footwear",
    "tie": "accessory",
    "scarf": "accessory",
    "hat": "accessory",
}

SCENE_KEYWORDS = {
    "office": ["office", "boardroom", "conference room", "workplace"],
    "street": ["street", "city walk", "sidewalk", "crosswalk", "city"],
    "park": ["park", "park bench", "outdoors"],
    "home": ["home", "couch", "living room", "indoors"],
}

STYLE_KEYWORDS = {
    "formal": ["formal"],
    "business": ["business", "professional"],
    "casual": ["casual", "weekend"],
    "athleisure": ["athleisure", "athletic", "gym", "workout"],
}

# multi-word garment types must be checked before single-word ones
_GARMENT_TYPES_BY_LENGTH = sorted(GARMENT_SLOTS, key=len, reverse=True)


def _find_scene(query_lower: str) -> str | None:
    for scene, keywords in SCENE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return scene
    return None


def _find_style(query_lower: str) -> str | None:
    for style, keywords in STYLE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return style
    return None


def _find_garments(query_lower: str) -> list[Garment]:
    garments: list[Garment] = []
    seen_slots: set[str] = set()

    for garment_type in _GARMENT_TYPES_BY_LENGTH:
        match = re.search(rf"\b{re.escape(garment_type)}\b", query_lower)
        if not match:
            continue

        slot = GARMENT_SLOTS[garment_type]
        if slot in seen_slots:
            continue

        preceding_text = query_lower[: match.start()]
        preceding_words = re.findall(r"\w+", preceding_text)
        color = next(
            (w for w in reversed(preceding_words[-3:]) if w in COLORS),
            None,
        )

        garments.append(Garment(slot=slot, type=garment_type, color=color))
        seen_slots.add(slot)

    return garments


def parse_query_keywords(raw_query: str) -> ParsedQuery:
    query_lower = raw_query.lower()

    garments = _find_garments(query_lower)
    scene = _find_scene(query_lower)
    style = _find_style(query_lower)

    matched_signal_count = len(garments) + (scene is not None) + (style is not None)
    confidence = min(1.0, 0.4 + 0.2 * matched_signal_count)

    return ParsedQuery(
        raw_query=raw_query,
        garments=garments,
        scene=scene,
        style=style,
        confidence=confidence,
    )
