"""One-off script: zero-shot CLIP classification of scene and style for the
40 real Fashionpedia records, per Working_notes.md Section 4.1 step 4 --
"Fashionpedia doesn't natively label environment [or style] -- tag it
yourself via zero-shot CLIP classification".

NOT applied to the shipped data. Tried this against the real images and
found it unreliable: rewording the scene prompts (still reasonable,
visually-grounded wording) flipped the predicted distribution from
"office"-dominant (18/40) to "park"-dominant (30/40) on the exact same
images, and the same test on style prompts collapsed from a spread across
all 4 labels to "formal"/"casual" only, with "business" never winning
despite being an option both times. Confidence margins throughout are
razor-thin (~0.17-0.29 raw cosine similarity) regardless of wording --
this dataset's fashion-photography backgrounds are often deliberately
blurred to keep focus on the garment, so there may just not be much
scene signal for CLIP to find. See Working_notes.md Section 12.3 for the
full writeup. Kept in the repo as a documented negative result, not a
working solution -- a real fix would need prompt ensembling (the
original CLIP zero-shot recipe averages ~80 templates per class) or a
proper VLM call, not a single hand-written prompt per class.

Does NOT need a VLM API key: uses the same local open_clip model as
scripts/eval_clip_baseline.py, just for labeling instead of retrieval.
If you disagree with the finding above and want to use it anyway: it
mutates app/data/real_catalog_sample.json in place -- sets scene/style
(previously null) and appends them to each record's caption so the dense
scorer picks them up too. color stays null regardless; there's still no
signal for it without a real VLM/attribute call.

Run with: pip install -r scripts/requirements-eval.txt
          python scripts/tag_real_catalog_scene_style.py
Requires images already on disk (pull_fashionpedia_sample.py).
"""

import json
from pathlib import Path

import open_clip
import torch
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BACKEND_DIR / "app" / "data" / "fashionpedia_images"
CATALOG_PATH = BACKEND_DIR / "app" / "data" / "real_catalog_sample.json"

# label -> CLIP prompt. Labels match query_parser.py's SCENE_KEYWORDS /
# STYLE_KEYWORDS vocabulary exactly, so parsed queries actually match what
# gets written here.
SCENE_PROMPTS = {
    "office": "an indoor office interior with desks, computers, or fluorescent lighting",
    "street": "an outdoor city street with buildings, sidewalks, or urban architecture",
    "park": "an outdoor park or nature setting with grass, trees, or greenery",
    "home": "an indoor home interior such as a living room or bedroom",
}
STYLE_PROMPTS = {
    "formal": "a person wearing formal attire",
    "casual": "a person wearing casual attire",
    "athleisure": "a person wearing athletic or athleisure attire",
    "business": "a person wearing business attire",
}


def classify(image_features, model, tokenizer, prompts: dict[str, str]) -> tuple[str, float]:
    labels = list(prompts)
    text = tokenizer([prompts[label] for label in labels])
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    sims = (image_features @ text_features.T).squeeze(0)
    best_idx = sims.argmax().item()
    return labels[best_idx], sims[best_idx].item()


def main() -> None:
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.glob("fp_*.jpg")):
        raise SystemExit(f"No images found at {IMAGES_DIR}. Run pull_fashionpedia_sample.py first.")

    print("Loading CLIP (openai/ViT-B-32)...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    records = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    records_by_id = {r["id"]: r for r in records}

    print(f"{'id':<10} {'scene':<10} {'conf':<6} {'style':<12} {'conf':<6}")
    print("-" * 50)

    for path in sorted(IMAGES_DIR.glob("fp_*.jpg")):
        record = records_by_id.get(path.stem)
        if record is None:
            continue

        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        scene, scene_conf = classify(image_features, model, tokenizer, SCENE_PROMPTS)
        style, style_conf = classify(image_features, model, tokenizer, STYLE_PROMPTS)

        record["scene"] = scene
        record["style"] = style
        record["caption"] = f"{record['caption']}, {style} style, {scene} setting"

        print(f"{record['id']:<10} {scene:<10} {scene_conf:<6.2f} {style:<12} {style_conf:<6.2f}")

    CATALOG_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nUpdated {CATALOG_PATH}")


if __name__ == "__main__":
    main()
