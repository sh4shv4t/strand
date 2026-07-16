from fastapi import APIRouter, HTTPException

from app.services.indexer import IndexingDependenciesMissing, index_image

router = APIRouter(prefix="/api", tags=["index"])


@router.post("/index")
def index(image_path: str) -> dict[str, object]:
    """Part A: The Indexer, exposed over the API. Runs real feature
    extraction (services/indexer.py, CLIP-based, no VLM key needed) and
    persists the embedding. Returns 503 if the heavy optional
    dependencies (torch, open_clip) are not installed in this
    deployment, 404 if image_path does not exist.
    """
    try:
        embedding = index_image(image_path)
    except IndexingDependenciesMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"status": "indexed", "embedding_dim": len(embedding)}
