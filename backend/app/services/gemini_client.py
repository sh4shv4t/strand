"""Shared Gemini client wrapper. Both real-model call sites in this repo
(llm_query_parser.py at query time, vlm_attribute_extractor.py at index
time) go through generate_structured() here rather than talking to the
SDK directly, so there is exactly one place that knows how to build a
client, one place that decides which model to call, and one place that
turns "no API key" into a clear, specific exception instead of a raw SDK
error surfacing wherever the call happened to be made.

Needs GEMINI_API_KEY set (see .env.example). Nothing in this module runs
at import time; the client is created lazily on first real use, so the
app, tests, and CI are unaffected when no key is configured.
"""

import os

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

_client = None


class GeminiNotConfigured(RuntimeError):
    pass


def get_client():
    global _client
    if _client is not None:
        return _client

    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiNotConfigured(
            "GEMINI_API_KEY is not set. Copy backend/.env.example to backend/.env "
            "and fill it in, or export it directly."
        )

    try:
        _client = genai.Client(api_key=api_key)
    except ValueError as exc:
        # The SDK itself raises ValueError for a missing/empty key; wrap it
        # in our own exception so callers only need to catch one type.
        raise GeminiNotConfigured(str(exc)) from exc

    return _client


def generate_structured(*, contents, response_schema, system_instruction: str | None = None, model: str | None = None):
    """Calls Gemini with structured JSON output and returns a validated
    instance of response_schema (a Pydantic model class), not raw text.

    Raises GeminiNotConfigured if no API key is set, or
    google.genai.errors.APIError for any actual API-level failure (auth,
    rate limit, server error); callers decide whether to fall back.
    """
    from google.genai import types

    client = get_client()

    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction,
        ),
    )
    return response.parsed
