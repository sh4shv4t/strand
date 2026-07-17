"""Query parsing entry point. routers/query.py calls parse_query() here,
not either implementation directly: this is the one place that decides
between the real LLM parser and the keyword-spotting fallback, so that
decision doesn't leak into the API layer or into either parser.

Tries the real LLM parser (llm_query_parser.py) first. Falls back to the
keyword parser (query_parser.py) if Gemini is not configured, or if the
call fails for any reason (network, rate limit, malformed response) --
a broken or absent API key degrades the app to its previous behavior
rather than making it error out.
"""

from app.observability import logger
from app.schema import ParsedQuery
from app.services.gemini_client import GeminiNotConfigured
from app.services.llm_query_parser import parse_query_llm
from app.services.query_parser import parse_query_keywords


def parse_query(raw_query: str) -> ParsedQuery:
    try:
        return parse_query_llm(raw_query)
    except GeminiNotConfigured:
        return parse_query_keywords(raw_query)
    except Exception:
        logger.exception("LLM query parsing failed, falling back to keyword parser")
        return parse_query_keywords(raw_query)
