from fastapi import APIRouter, HTTPException

from app.services.indexer import index_image

router = APIRouter(prefix="/api", tags=["index"])


@router.post("/index")
def index(image_path: str) -> None:
    """Stub endpoint for the offline indexing loop (Working_notes.md Section 4.1).

    Not implemented in this scaffold -- always returns 501. Exists so the
    API surface for the eventual real indexing pipeline is visible now.
    """
    try:
        index_image(image_path)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
