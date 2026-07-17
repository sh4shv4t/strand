"""Real CLIP image-embedding similarity: the query-time counterpart to
indexer.py's index-time feature extraction.

This is what actually wires Part A's real image embeddings into
retrieval. Before this module existed, index_image() persisted a real
CLIP embedding for every real Fashionpedia photo into image_vector_store,
and nothing ever read it back, retriever.py's "dense" signal was only
ever a text embedding of each record's auto-generated caption, never the
image pixels themselves. score() encodes the query text with the SAME
CLIP model's text tower (clip_model.py), landing in the same joint
embedding space as the stored image vectors, then compares it against
every stored embedding via cosine similarity.

Returns {} (not an error) when torch/open_clip aren't installed, the
image vector store is empty, or STRAND_DISABLE_EMBEDDINGS is set.
retriever.py treats a missing id here exactly like a record with no
image on file (the 12 hand-written mock records have none): falls back
to caption-only dense similarity for it, rather than failing the whole
query. Respects STRAND_DISABLE_EMBEDDINGS for the same reason
vector_store.py does: CI never has torch/open_clip installed at all
(intentionally excluded from requirements.txt, see clip_model.py), and
local test runs shouldn't pay for a real model load either.
"""

import os

from app.services.clip_model import ClipDependenciesMissing, get_model_and_preprocess, get_tokenizer
from app.services.image_vector_store import get_image_collection

_DISABLE_EMBEDDINGS_ENV = "STRAND_DISABLE_EMBEDDINGS"


def _disabled() -> bool:
    return os.environ.get(_DISABLE_EMBEDDINGS_ENV, "").lower() in ("1", "true")


def score(raw_query: str) -> dict[str, float]:
    """Real image-pixel similarity for every id with a stored CLIP
    embedding: id -> similarity in [0, 1]. Empty dict if unavailable for
    any reason, see module docstring.
    """
    if _disabled():
        return {}

    try:
        model, _ = get_model_and_preprocess()
        tokenizer = get_tokenizer()
    except ClipDependenciesMissing:
        return {}

    collection = get_image_collection()
    count = collection.count()
    if count == 0:
        return {}

    import torch

    with torch.no_grad():
        text_features = model.encode_text(tokenizer([raw_query]))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    result = collection.query(
        query_embeddings=[text_features.squeeze(0).tolist()],
        n_results=count,
    )
    ids = result["ids"][0]
    distances = result["distances"][0]
    # cosine space (see image_vector_store.py's hnsw:space metadata):
    # distance = 1 - cosine_similarity, same convention as vector_store.py.
    return {rid: max(0.0, min(1.0, 1.0 - dist)) for rid, dist in zip(ids, distances)}
