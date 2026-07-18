"""Query parsing entry point. routers/query.py calls parse_query() here,
not either implementation directly: this is the one place that decides
between the real LLM parser and the keyword-spotting fallback, so that
decision doesn't leak into the API layer or into either parser.

Tries the real LLM parser (llm_query_parser.py) first. Falls back to the
keyword parser (query_parser.py) if Gemini is not configured, or if the
call fails for any reason (network, rate limit, malformed response) --
a broken or absent API key degrades the app to its previous behavior
rather than making it error out.

parse_query() is cached (see _parse_query_cached): re-parsing an
identical query string, the example chips, anything a user re-runs,
gains nothing, an LLM call is real latency and real cost, and even the
keyword parser's regex work is pure waste to redo. Accepted, narrow
tradeoff: if a transient failure trips the LLM fallback once (a network
blip), the degraded keyword-parser result gets cached for that exact
query text until evicted, a retry moments later that would have
succeeded against Gemini won't happen automatically. Given repeats of
the exact same failure window are unlikely, this is a reasonable price
for not re-parsing on every repeat query.
"""

from functools import lru_cache

from app.observability import logger
from app.schema import ParsedQuery
from app.services.gemini_client import GeminiNotConfigured
from app.services.llm_query_parser import parse_query_llm
from app.services.query_parser import parse_query_keywords


def parse_query(raw_query: str) -> ParsedQuery:
    # A copy, not the cached object itself: ParsedQuery is a mutable
    # Pydantic model, callers must not be able to corrupt what every
    # future identical query gets back.
    return _parse_query_cached(raw_query).model_copy(deep=True)


@lru_cache(maxsize=256)
def _parse_query_cached(raw_query: str) -> ParsedQuery:
    try:
        return parse_query_llm(raw_query)
    except GeminiNotConfigured:
        return parse_query_keywords(raw_query)
    except Exception:
        logger.exception("LLM query parsing failed, falling back to keyword parser")
        return parse_query_keywords(raw_query)


def clear_cache() -> None:
    """Nothing in this module's own state needs this today, parsing
    doesn't depend on catalog contents, exposed for tests and for
    symmetry with the other cached score functions."""
    _parse_query_cached.cache_clear()
