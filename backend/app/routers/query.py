from fastapi import APIRouter

from app.schema import QueryRequest, QueryResponse
from app.services.query_parser import parse_query
from app.services.retriever import search

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    parsed = parse_query(request.query)
    results = search(parsed, top_k=request.top_k, alpha=request.alpha)
    return QueryResponse(parsed=parsed, results=results)
