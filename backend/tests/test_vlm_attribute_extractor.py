"""vlm_attribute_extractor.py tests. See test_gemini_client.py for why
these don't make a real API call."""

import pytest

from app.services.gemini_client import GeminiNotConfigured
from app.services.vlm_attribute_extractor import extract_attributes


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_attributes("this/path/does/not/exist.jpg")


def test_raises_when_not_configured(monkeypatch, tmp_path):
    import app.services.gemini_client as gemini_client_module

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_client_module, "_client", None)

    # GeminiNotConfigured fires from get_client() before generate_structured
    # ever sends the image anywhere, so the file just needs to exist, its
    # bytes are never inspected. No need for a real image / Pillow here.
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"not a real jpeg")

    with pytest.raises(GeminiNotConfigured):
        extract_attributes(str(image_path))
