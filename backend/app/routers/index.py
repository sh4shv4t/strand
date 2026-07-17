from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schema import Garment, ImageRecord, Scene, Style
from app.services.indexer import IndexingDependenciesMissing, index_image
from app.services.retriever import register_record

router = APIRouter(prefix="/api", tags=["index"])


class IndexRequest(BaseModel):
    image_path: str
    garments: list[Garment] = Field(default_factory=list)
    scene: Scene | None = None
    style: Style | None = None


@router.post("/index")
def index(request: IndexRequest) -> dict[str, object]:
    """Part A: The Indexer, exposed over the API. Runs real feature
    extraction (services/indexer.py, CLIP-based, no VLM key needed) and
    persists the embedding. Returns 503 if the heavy optional
    dependencies (torch, open_clip) are not installed in this
    deployment, 404 if image_path does not exist.

    garments/scene/style are optional. Without them this only persists a
    real CLIP embedding, Part A's literal requirement, same behavior as
    before this field existed. Supplying garments additionally registers
    a full record into the live catalog so the image is searchable via
    /api/query immediately: this is the cold-start fix. Previously,
    calling this endpoint wrote a real embedding into image_vector_store
    and nothing ever made the image reachable from a query, an image
    without a real color/scene/style call (still needs a VLM key,
    Working_notes.md Section 8) is exactly the case a newly onboarded
    image is in today, so it registers with whatever is supplied and
    null for the rest, the same honest-null convention the real
    Fashionpedia sample already uses.
    """
    try:
        embedding = index_image(request.image_path)
    except IndexingDependenciesMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    registered = False
    if request.garments:
        record_id = Path(request.image_path).stem
        caption = ", ".join(f"{g.type} ({g.slot})" for g in request.garments)
        record = ImageRecord(
            id=record_id,
            garments=request.garments,
            scene=request.scene,
            style=request.style,
            caption=caption,
        )
        register_record(record)
        registered = True

    return {"status": "indexed", "embedding_dim": len(embedding), "registered": registered}
