from fastapi import APIRouter

from app.observability import logger, tracer
from app.schema import QueryRequest, QueryResponse
from app.services.query_parser import parse_query
from app.services.retriever import search

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    with tracer.start_as_current_span("parse_query") as span:
        span.set_attribute("query.raw", request.query)
        parsed = parse_query(request.query)
        span.set_attribute("query.parsed_garment_count", len(parsed.garments))
        span.set_attribute("query.confidence", parsed.confidence)

    with tracer.start_as_current_span("retrieve") as span:
        results = search(parsed, top_k=request.top_k, alpha=request.alpha)
        span.set_attribute("retrieve.result_count", len(results))
        if results:
            span.set_attribute("retrieve.top_score", results[0].score)

    logger.info(
        "query=%r parsed_garments=%d top_result=%s top_score=%s",
        request.query,
        len(parsed.garments),
        results[0].record.id if results else None,
        results[0].score if results else None,
    )

    return QueryResponse(parsed=parsed, results=results)
