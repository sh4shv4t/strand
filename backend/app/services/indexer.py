"""Part A: The Indexer, real feature extraction from raw images.

Runs each image through a local CLIP model (open_clip, no API key needed)
to get a real image embedding, and persists it via image_vector_store.
This module used to be a documented stub; it now does the part of
"Feature Extraction... Vector Storage" (see Working_notes.md Section 4.1)
that is achievable with a local, keyless model: turning pixels into a
searchable vector representation, for real.

This deliberately does not touch color, scene, or style. That axis still
needs a real VLM call (Working_notes.md Section 4.2), which is not wired
up yet, and is a separate concern from feature extraction.

Heavy dependencies (torch, open_clip) are not in the app's own
requirements.txt, see scripts/requirements-eval.txt, so they are imported
lazily here. Calling index_image without them installed raises
IndexingDependenciesMissing with a clear message instead of failing at
import time for the whole app.
"""

from pathlib import Path

from app.services.image_vector_store import store_embedding

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"

_model = None
_preprocess = None


class IndexingDependenciesMissing(RuntimeError):
    pass


def _load_model():
    global _model, _preprocess
    if _model is not None:
        return _model, _preprocess

    try:
        import open_clip
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise IndexingDependenciesMissing(
            "Real feature extraction needs open_clip, torch, and pillow, which "
            "are not installed by default (they are heavy and only needed for "
            "indexing, not for serving queries). Install with: "
            "pip install -r scripts/requirements-eval.txt"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(_MODEL_NAME, pretrained=_PRETRAINED)
    model.eval()
    _model, _preprocess = model, preprocess
    return _model, _preprocess


def index_image(image_path: str) -> list[float]:
    """Extracts a real image embedding from image_path via CLIP, persists
    it to the image vector store keyed by the file's stem, and returns it.

    Raises IndexingDependenciesMissing if open_clip/torch/pillow are not
    installed, or FileNotFoundError if image_path does not exist.
    """
    model, preprocess = _load_model()

    import torch
    from PIL import Image

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image_path does not exist: {image_path}")

    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding /= embedding.norm(dim=-1, keepdim=True)

    vector = embedding.squeeze(0).tolist()
    store_embedding(path.stem, vector)
    return vector
