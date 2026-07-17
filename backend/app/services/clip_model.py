"""Shared local CLIP model (open_clip, ViT-B-32, openai weights, no API
key). Used by both indexer.py (image encoding, Part A) and
image_similarity.py (text encoding, to rank against those same
embeddings at query time) -- loading it once and sharing the instance
avoids holding two copies of the same model in memory in one process.

Heavy dependencies (torch, open_clip, pillow) are not in the app's own
requirements.txt, see scripts/requirements-eval.txt, so they are imported
lazily here. Calling get_model_and_preprocess()/get_tokenizer() without
them installed raises ClipDependenciesMissing with a clear message
instead of failing at import time for the whole app.
"""

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

_model = None
_preprocess = None
_tokenizer = None


class ClipDependenciesMissing(RuntimeError):
    pass


def get_model_and_preprocess():
    global _model, _preprocess
    if _model is not None:
        return _model, _preprocess

    try:
        import open_clip
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise ClipDependenciesMissing(
            "Real CLIP encoding needs open_clip, torch, and pillow, which "
            "are not installed by default (they are heavy and only needed "
            "for indexing/ranking against real image embeddings, not for "
            "basic query serving). Install with: "
            "pip install -r scripts/requirements-eval.txt"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    _model, _preprocess = model, preprocess
    return _model, _preprocess


def get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    try:
        import open_clip
    except ImportError as exc:
        raise ClipDependenciesMissing(
            "Real CLIP encoding needs open_clip, torch, and pillow, which "
            "are not installed by default (they are heavy and only needed "
            "for indexing/ranking against real image embeddings, not for "
            "basic query serving). Install with: "
            "pip install -r scripts/requirements-eval.txt"
        ) from exc

    _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return _tokenizer
