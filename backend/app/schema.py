"""JSON schema shared by index-time records and query-time parses.

Mirrors the schema proposed in Working_notes.md Section 2, Option D:
garments are bound to a slot (upper/lower/outerwear/footwear/accessory)
so that "red tie, white shirt" cannot be satisfied by a white tie and a
red shirt — the compositional failure mode vanilla CLIP has.
"""

from typing import Optional

from pydantic import BaseModel, Field

Slot = str  # "upper" | "lower" | "outerwear" | "footwear" | "accessory"
Scene = str  # "office" | "street" | "park" | "home" | "other"
Style = str  # "formal" | "casual" | "athleisure" | "business" | "other"


class Garment(BaseModel):
    slot: Slot
    type: str
    color: Optional[str] = None


class ExtractedAttributes(BaseModel):
    """Garments/scene/style shape shared by both Gemini call sites: the
    query-time parser (llm_query_parser.py) extracts this from query text,
    the index-time extractor (vlm_attribute_extractor.py) extracts the
    same shape from an image. Used directly as the Gemini response_schema
    so the SDK returns a validated instance of this model, not raw JSON.
    """

    garments: list[Garment] = Field(default_factory=list)
    scene: Optional[Scene] = None
    style: Optional[Style] = None


class ImageRecord(BaseModel):
    id: str
    garments: list[Garment]
    scene: Optional[Scene] = None
    style: Optional[Style] = None
    notable: list[str] = Field(default_factory=list)
    caption: str
    swatch: list[str] = Field(default_factory=list)
    # Slots whose color came from color_detection.py's RGB read rather
    # than a hand-authored or VLM-confirmed source. Kept off Garment
    # itself (and off ExtractedAttributes/ParsedQuery, which Garment is
    # also shared with) specifically so it never becomes a field the
    # Gemini structured-output schema asks the LLM to fill in, this is
    # bookkeeping the retriever consults, not something to extract from
    # a query. retriever.py uses this to treat a detected color as a
    # soft signal instead of the hard symbolic gate a confident color
    # gets, see Working_notes.md Section 17.
    detected_color_slots: list[str] = Field(default_factory=list)


class ParsedQuery(BaseModel):
    raw_query: str
    garments: list[Garment] = Field(default_factory=list)
    scene: Optional[Scene] = None
    style: Optional[Style] = None
    confidence: float = 1.0


class ScoredResult(BaseModel):
    record: ImageRecord
    score: float
    symbolic_score: float
    dense_score: float
    matched_fields: list[str]


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    parsed: ParsedQuery
    results: list[ScoredResult]
