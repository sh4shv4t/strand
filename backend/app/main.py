from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import catalog, index, query

app = FastAPI(
    title="Strand API",
    description="Every detail, connected. Mock retrieval driver for the Glance ML internship assignment.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(catalog.router)
app.include_router(index.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
