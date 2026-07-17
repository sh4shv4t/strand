"""Tier 2 baseline: a real vanilla-CLIP image embedding (Option A from
Working_notes.md Section 2, literally -- whole-image embedding, zero-shot
text-image similarity) against the 40 real Fashionpedia photos, compared
to our hybrid system and its dense-only baseline on the SAME real images
and the SAME ground truth (Fashionpedia's own category labels).

Probe queries are single-garment-type queries built from types that both
(a) appear at least 3x in the real sample, and (b) are recognized by
query_parser.py's current keyword vocabulary, so the symbolic layer
actually gets a chance to engage for all three methods being compared.

Run with: pip install open_clip_torch && python scripts/eval_clip_baseline.py
Requires images from `pull_fashionpedia_sample.py` to already be on disk
at app/data/fashionpedia_images/.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STRAND_DISABLE_EMBEDDINGS", "1")

import open_clip
import torch
from PIL import Image

from app.services.catalog import get_catalog
# Uses the keyword parser directly, not query_parsing.parse_query, so this
# baseline comparison stays deterministic and free regardless of whether a
# GEMINI_API_KEY happens to be set in the environment it runs in.
from app.services.query_parser import parse_query_keywords as parse_query
from app.services.retriever import search

BACKEND_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BACKEND_DIR / "app" / "data" / "fashionpedia_images"
CATALOG_PATH = BACKEND_DIR / "app" / "data" / "real_catalog_sample.json"

PROBE_QUERIES = [
    ("a pair of shoes", "shoe"),
    ("a dress", "dress"),
    ("a t-shirt", "t-shirt"),
    ("pants", "pants"),
    ("shorts", "shorts"),
    ("a jacket", "jacket"),
    ("a shirt", "shirt"),
    ("a skirt", "skirt"),
]

TOP_K = 5


def ground_truth(garment_type: str) -> set[str]:
    records = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {r["id"] for r in records if any(g["type"] == garment_type for g in r["garments"])}


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def our_system_topk(query: str, alpha: float) -> list[str]:
    parsed = parse_query(query)
    results = search(parsed, top_k=len(get_catalog()), alpha=alpha)
    real_ids = [r.record.id for r in results if r.record.id.startswith("fp_")]
    return real_ids[:TOP_K]


def build_clip_image_index():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    image_ids, images = [], []
    for path in sorted(IMAGES_DIR.glob("fp_*.jpg")):
        image_ids.append(path.stem)
        images.append(preprocess(Image.open(path).convert("RGB")))

    batch = torch.stack(images)
    with torch.no_grad():
        image_features = model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return model, tokenizer, image_ids, image_features


def clip_topk(model, tokenizer, image_ids, image_embeds, query: str) -> list[str]:
    text = tokenizer([query])
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    sims = (image_embeds @ text_features.T).squeeze(1)
    ranked_idx = sims.argsort(descending=True)
    return [image_ids[i] for i in ranked_idx[:TOP_K].tolist()]


def main() -> None:
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.glob("fp_*.jpg")):
        raise SystemExit(
            f"No images found at {IMAGES_DIR}. Run pull_fashionpedia_sample.py first."
        )

    print("Loading CLIP (openai/ViT-B-32) and embedding real images...")
    model, tokenizer, image_ids, image_embeds = build_clip_image_index()
    print(f"Embedded {len(image_ids)} real Fashionpedia images.\n")

    header = f"{'probe query':<18} {'#relevant':<10} {'CLIP r@5':<10} {'dense-only r@5':<16} {'hybrid r@5':<12}"
    print(header)
    print("-" * len(header))

    clip_recalls, dense_recalls, hybrid_recalls = [], [], []

    for query, garment_type in PROBE_QUERIES:
        relevant = ground_truth(garment_type)
        if not relevant:
            continue

        clip_ids = clip_topk(model, tokenizer, image_ids, image_embeds, query)
        dense_ids = our_system_topk(query, alpha=0.0)
        hybrid_ids = our_system_topk(query, alpha=0.6)

        clip_r = recall_at_k(clip_ids, relevant, TOP_K)
        dense_r = recall_at_k(dense_ids, relevant, TOP_K)
        hybrid_r = recall_at_k(hybrid_ids, relevant, TOP_K)

        clip_recalls.append(clip_r)
        dense_recalls.append(dense_r)
        hybrid_recalls.append(hybrid_r)

        print(f"{query:<18} {len(relevant):<10} {clip_r:<10.2f} {dense_r:<16.2f} {hybrid_r:<12.2f}")

    print("-" * len(header))
    print(
        f"mean recall@5 -- CLIP: {sum(clip_recalls) / len(clip_recalls):.3f}  "
        f"dense-only: {sum(dense_recalls) / len(dense_recalls):.3f}  "
        f"hybrid: {sum(hybrid_recalls) / len(hybrid_recalls):.3f}"
    )


if __name__ == "__main__":
    main()
