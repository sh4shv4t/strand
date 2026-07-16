import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.observability import logger, tracer
from app.routers import catalog, index, query
from app.services.catalog import get_catalog
from app.services.retriever import get_dense_scorer

app = FastAPI(
    title="Strand API",
    description="Every detail, connected. Mock retrieval driver for the Glance ML internship assignment.",
    version="0.1.0",
)

cors_origins = os.environ.get("STRAND_CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Real Fashionpedia photos, if pulled (pull_fashionpedia_sample.py) -- gitignored
# and optional, so only mount if present; the frontend degrades gracefully
# (falls back to swatch-only cards) when a photo 404s.
_images_dir = Path(__file__).resolve().parent / "data" / "fashionpedia_images"
if _images_dir.exists():
    app.mount("/api/images", StaticFiles(directory=_images_dir), name="images")


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        span.set_attribute("http.method", request.method)
        span.set_attribute("http.route", request.url.path)
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("http.duration_ms", duration_ms)

        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )


app.include_router(query.router)
app.include_router(catalog.router)
app.include_router(index.router)


@app.get("/api/health")
def health() -> dict[str, object]:
    catalog = get_catalog()
    if not catalog:
        raise HTTPException(status_code=503, detail="catalog failed to load (empty)")
    return {
        "status": "ok",
        "catalog_size": len(catalog),
        "using_real_embeddings": get_dense_scorer().using_real_embeddings,
    }
