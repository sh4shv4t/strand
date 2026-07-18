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

score() is cached (see _score_cached): encoding the query text through
CLIP is the real cost here, a repeated query should not pay for it
twice. The collection's current size is folded into the cache key, so a
newly indexed image (which changes that count) naturally busts stale
entries without a separate invalidation call, unlike DenseScorer, this
module needs no explicit registration step (module docstring above).
This doesn't catch every possible staleness case, re-indexing an
existing image with a new embedding without changing the collection's
size would still return a stale cached result, an accepted, narrow
tradeoff, not a claim of perfect invalidation.
"""

import os
from functools import lru_cache

from app.services.clip_model import ClipDependenciesMissing, get_model_and_preprocess, get_tokenizer
from app.services.image_vector_store import get_image_collection

_DISABLE_EMBEDDINGS_ENV = "STRAND_DISABLE_EMBEDDINGS"


def _disabled() -> bool:
    return os.environ.get(_DISABLE_EMBEDDINGS_ENV, "").lower() in ("1", "true")


def score(raw_query: str) -> dict[str, float]:
    """Real image-pixel similarity for every id with a stored CLIP
    embedding: id -> similarity in [0, 1]. Empty dict if unavailable for
    any reason, see module docstring. Checked outside the cache so
    toggling STRAND_DISABLE_EMBEDDINGS (tests do this per-test) is never
    masked by a cached result from a different setting.
    """
    if _disabled():
        return {}

    try:
        get_model_and_preprocess()
        get_tokenizer()
    except ClipDependenciesMissing:
        return {}

    collection = get_image_collection()
    count = collection.count()
    if count == 0:
        return {}

    # A copy, not the cached dict itself: callers must not be able to
    # mutate what every future identical query gets back.
    return dict(_score_cached(raw_query, count))


@lru_cache(maxsize=256)
def _score_cached(raw_query: str, count: int) -> dict[str, float]:
    # model/tokenizer/collection are deliberately re-fetched here rather
    # than passed in as arguments: they're stable singletons for the life
    # of the process (clip_model.py, image_vector_store.py), so they add
    # nothing to cache correctness, and Chroma's Collection object isn't
    # hashable, so passing it as a cache key argument would crash the
    # very first real (non-disabled) call. raw_query and count are the
    # only two things that actually need to vary the cache key.
    model, _ = get_model_and_preprocess()
    tokenizer = get_tokenizer()
    collection = get_image_collection()

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


def clear_cache() -> None:
    """Not wired into retriever.register_record(): unlike DenseScorer,
    nothing register_record does changes what this module reads (see
    module docstring). Exposed for tests and for symmetry with
    vector_store.DenseScorer.clear_cache()."""
    _score_cached.cache_clear()
