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


class ImageRecord(BaseModel):
    id: str
    garments: list[Garment]
    scene: Optional[Scene] = None
    style: Optional[Style] = None
    notable: list[str] = Field(default_factory=list)
    caption: str
    swatch: list[str] = Field(default_factory=list)


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
    top_k: int = 5
    alpha: float = 0.6


class QueryResponse(BaseModel):
    parsed: ParsedQuery
    results: list[ScoredResult]
