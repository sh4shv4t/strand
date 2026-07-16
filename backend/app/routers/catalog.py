from fastapi import APIRouter

from app.schema import ImageRecord
from app.services.retriever import get_catalog

router = APIRouter(prefix="/api", tags=["catalog"])


@router.get("/catalog", response_model=list[ImageRecord])
def catalog() -> list[ImageRecord]:
    return get_catalog()
