"""Persistent Chroma collection for real image embeddings (Part A: Vector
Storage). Separate from vector_store.DenseScorer's caption-embedding
collection, which is ephemeral and used for live query serving: this one
is backed by real pixels via indexer.index_image, and persists to disk
so the vector store survives a restart, not just the lifetime of one
process.
"""

from pathlib import Path

import chromadb

PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "image_vector_index"

_client = None
_collection = None


def get_image_collection():
    global _client, _collection
    if _collection is None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        _collection = _client.get_or_create_collection(
            "strand_image_vectors", metadata={"hnsw:space": "cosine"}
        )
    return _collection


def store_embedding(record_id: str, embedding: list[float]) -> int:
    """Upserts one image's embedding. Returns the collection's new size."""
    collection = get_image_collection()
    collection.upsert(ids=[record_id], embeddings=[embedding])
    return collection.count()
