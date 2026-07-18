"""Dense similarity scoring, backed by a local Chroma collection.

Isolated from retriever.py so it can be swapped or mocked independently:
retriever.py only depends on `DenseScorer.score(query) -> {id: similarity}`,
not on Chroma itself.

score() is cached (see _score_cached): encoding the query text is the
real cost here, not looking it up, so a repeated query (the example
chips, anything a user re-runs) should not pay to re-embed. Cached
per-record results depend on which records are in the collection at the
time, so retriever.register_record() clears this cache (clear_cache())
whenever a new record is added, otherwise a query cached before that
addition would keep returning a result missing the new record.
"""

import os
import re
from functools import lru_cache

import chromadb

from app.schema import ImageRecord

_WORD_RE = re.compile(r"\w+")

# Tests set this to skip the real embedding download entirely, so CI stays
# fast and network-independent -- they exercise the (deterministic) word
# overlap fallback path instead. Unset/false in normal operation.
_DISABLE_EMBEDDINGS_ENV = "STRAND_DISABLE_EMBEDDINGS"


def _word_overlap_score(raw_query: str, caption: str) -> float:
    query_words = set(_WORD_RE.findall(raw_query.lower()))
    caption_words = set(_WORD_RE.findall(caption.lower()))
    if not query_words or not caption_words:
        return 0.0
    overlap = query_words & caption_words
    return len(overlap) / len(query_words | caption_words)


class DenseScorer:
    """Real cosine similarity via Chroma's default local embedding function
    (no API key, no external service). Falls back to word overlap between
    the query and each record's caption if the embedding model can't be
    loaded -- e.g. no network yet to cache its weights on first run -- so
    the app degrades instead of failing to start.
    """

    def __init__(self, records: list[ImageRecord], use_embeddings: bool | None = None):
        self._records = records
        if use_embeddings is None:
            use_embeddings = os.environ.get(_DISABLE_EMBEDDINGS_ENV, "").lower() not in (
                "1",
                "true",
            )
        self._collection = self._build_collection(records) if use_embeddings else None

    @staticmethod
    def _build_collection(records: list[ImageRecord]):
        if not records:
            return None
        try:
            client = chromadb.Client()
            collection = client.get_or_create_collection(
                "strand_catalog", metadata={"hnsw:space": "cosine"}
            )
            collection.add(
                ids=[r.id for r in records],
                documents=[r.caption for r in records],
            )
            return collection
        except Exception:
            return None

    @property
    def using_real_embeddings(self) -> bool:
        return self._collection is not None

    def add_record(self, record: ImageRecord) -> None:
        """Adds one record to the live collection so it is immediately
        dense-searchable. self._records is not touched here: callers
        (retriever.register_record) already share that same list object
        with _CATALOG, so it is already up to date by the time this
        runs. No-op when real embeddings are disabled, the word-overlap
        fallback re-scans self._records directly on every call, nothing
        to update ahead of time.
        """
        if self._collection is not None:
            self._collection.add(ids=[record.id], documents=[record.caption])

    def score(self, raw_query: str) -> dict[str, float]:
        # A copy, not the cached dict itself: callers must not be able to
        # mutate what every future identical query gets back.
        return dict(self._score_cached(raw_query))

    @lru_cache(maxsize=256)
    def _score_cached(self, raw_query: str) -> dict[str, float]:
        if self._collection is not None:
            try:
                result = self._collection.query(
                    query_texts=[raw_query], n_results=len(self._records)
                )
                ids = result["ids"][0]
                distances = result["distances"][0]
                # cosine space: distance = 1 - cosine_similarity
                return {
                    rid: max(0.0, min(1.0, 1.0 - dist)) for rid, dist in zip(ids, distances)
                }
            except Exception:
                pass

        return {record.id: _word_overlap_score(raw_query, record.caption) for record in self._records}

    def clear_cache(self) -> None:
        """Called by retriever.register_record() whenever a new record is
        added, a query cached before that would otherwise keep returning
        a result missing the addition."""
        self._score_cached.cache_clear()
