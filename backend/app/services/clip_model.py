"""Shared local CLIP model (open_clip, no API key). Used by both
indexer.py (image encoding, Part A) and image_similarity.py (text
encoding, to rank against those same embeddings at query time) --
loading it once and sharing the instance avoids holding two copies of
the same model in memory in one process.

Backbone is Marqo-FashionCLIP (Apache-2.0, loaded via open_clip's
hf-hub integration), not vanilla ViT-B-32/openai. Measured directly
against the real 1,000-image catalog on the same 8 probe queries Tier 2
(Working_notes.md Section 12.2) already used: vanilla CLIP scores 0.725
mean precision@5, Marqo-FashionCLIP scores a perfect 1.000, a fashion-
tuned backbone rather than a generic one, exactly the axis the
assignment's own hint says vanilla CLIP is weak on. Same 512-dim output
as ViT-B-32/openai, so this is a drop-in swap: no schema change, no
change to any persisted-embedding dimension assumption elsewhere.
scripts/eval_clip_baseline.py intentionally still hardcodes vanilla
ViT-B-32/openai directly, not this module, since that script's whole
purpose is comparing against vanilla CLIP as the baseline, not against
whatever backbone the production system happens to use.

Heavy dependencies (torch, open_clip, pillow) are not in the app's own
requirements.txt, see scripts/requirements-eval.txt, so they are imported
lazily here. Calling get_model_and_preprocess()/get_tokenizer() without
them installed raises ClipDependenciesMissing with a clear message
instead of failing at import time for the whole app.
"""

MODEL_NAME = "hf-hub:Marqo/marqo-fashionCLIP"
PRETRAINED = None

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
