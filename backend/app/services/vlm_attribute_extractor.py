"""Real VLM attribute extraction from an image: the index-time counterpart
to llm_query_parser.py's query-time extraction. Same ExtractedAttributes
schema on both sides of the parser (Working_notes.md Section 4.1 step 1
and Section 4.3), so an image and a query land in the same shape and the
retriever never has to reconcile two different structures.

This is the piece that fills in color, scene, and style on the real
Fashionpedia records, which are honestly null today (see
scripts/pull_fashionpedia_sample.py). It is separate from indexer.py's
CLIP-based feature extraction: that produces a dense vector from pixels,
this produces the structured schema from pixels. Both are real, local
where possible, and read pixels; only this one needs an API key, since
extracting a fixed vocabulary (garment slot/type/color/scene/style) from
an image is exactly the kind of open-vocabulary task a VLM is suited for
and CLIP alone is not (see Working_notes.md Section 12.3 for why a
zero-shot CLIP classification attempt at this specific task did not hold
up).

Needs GEMINI_API_KEY; raises GeminiNotConfigured if not set.
"""

from pathlib import Path

from app.schema import ExtractedAttributes
from app.services.gemini_client import generate_structured

SYSTEM_INSTRUCTION = """You are a fashion attribute extractor. Given a photo, identify every
visible garment (each bound to a slot: upper, lower, outerwear,
footwear, or accessory, with a type and a color), the scene the photo
was taken in (office, street, park, home, or other), and the overall
style (formal, casual, athleisure, business, or other).

Only include a garment, scene, or style you can actually see evidence
of in the image. Do not guess."""


def extract_attributes(image_path: str) -> ExtractedAttributes:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image_path does not exist: {image_path}")

    from google.genai import types

    image_part = types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg")

    return generate_structured(
        contents=[image_part, "Extract this photo's garments, scene, and style."],
        response_schema=ExtractedAttributes,
        system_instruction=SYSTEM_INSTRUCTION,
    )
