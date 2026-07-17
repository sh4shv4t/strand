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

Model loading itself lives in clip_model.py, shared with
image_similarity.py's query-time text encoding, so both sides of Part
A/B's real-CLIP integration use the same loaded model instead of two
copies of it in memory.
"""

from pathlib import Path

from app.services.clip_model import ClipDependenciesMissing as IndexingDependenciesMissing
from app.services.clip_model import get_model_and_preprocess
from app.services.image_vector_store import store_embedding


def index_image(image_path: str) -> list[float]:
    """Extracts a real image embedding from image_path via CLIP, persists
    it to the image vector store keyed by the file's stem, and returns it.

    Raises IndexingDependenciesMissing if open_clip/torch/pillow are not
    installed, or FileNotFoundError if image_path does not exist.
    """
    model, preprocess = get_model_and_preprocess()

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
