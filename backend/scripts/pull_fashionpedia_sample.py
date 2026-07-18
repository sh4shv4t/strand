"""One-off script: pull a small real sample from Fashionpedia and write it
to app/data/real_catalog_sample.json in the Strand schema. Also persists
the actual JPEGs to app/data/fashionpedia_images/ (gitignored -- rerun this
script to regenerate them; they're not committed) for the Tier 2 real-image
CLIP baseline in scripts/eval_clip_baseline.py.

Not part of the running app -- run manually to (re)generate the sample:

    pip install datasets pillow
    python scripts/pull_fashionpedia_sample.py

Uses the dataset's own ground-truth category labels (bbox category,
mapped to our garment slot/type taxonomy) for garment/slot detection, and
this dataset's own ground-truth bounding boxes for color: no VLM call
needed for that axis, services/color_detection.py crops each matched
garment's real bounding box and reads its actual dominant pixel color,
mapped to the nearest name in colors.COLOR_HEX. This HF mirror of
Fashionpedia (`detection-datasets/fashionpedia`) exposes bbox + category
only, not the original paper's 294 fine-grained attributes, so scene and
style still stay null (Working_notes.md Section 4.1 step 4: environment/
style tagging is a separate zero-shot/VLM step, deliberately deferred
until a VLM API key or local model is wired up), color does not have to.

Streaming iteration order is deterministic (no shuffling), so rerunning
this reproduces the same image_ids as the committed real_catalog_sample.json.

TARGET_COUNT is 1000 (the top of the assignment's stated 500-1,000 image
range): an earlier pass ran this at 40 for fast local iteration during
scaffolding, and that smaller sample was never re-pulled at the size the
assignment actually asks for before this.
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.color_detection import detect_color  # noqa: E402

TARGET_COUNT = 1000
IMAGES_DIR = Path(__file__).resolve().parent.parent / "app" / "data" / "fashionpedia_images"

# Fashionpedia category name -> (slot, our type name). Categories not listed
# here are garment *parts*/decorative details (collar, sleeve, zipper, bead,
# etc.) rather than whole garments, and are skipped.
SLOT_MAP = {
    "shirt, blouse": ("upper", "shirt"),
    "top, t-shirt, sweatshirt": ("upper", "t-shirt"),
    "sweater": ("upper", "sweater"),
    "dress": ("upper", "dress"),
    "jumpsuit": ("upper", "jumpsuit"),
    "cardigan": ("outerwear", "cardigan"),
    "jacket": ("outerwear", "jacket"),
    "vest": ("outerwear", "vest"),
    "coat": ("outerwear", "coat"),
    "cape": ("outerwear", "cape"),
    "pants": ("lower", "pants"),
    "shorts": ("lower", "shorts"),
    "skirt": ("lower", "skirt"),
    "shoe": ("footwear", "shoe"),
    "hat": ("accessory", "hat"),
    "tie": ("accessory", "tie"),
    "glove": ("accessory", "glove"),
    "watch": ("accessory", "watch"),
    "belt": ("accessory", "belt"),
    "bag, wallet": ("accessory", "bag"),
    "scarf": ("accessory", "scarf"),
    "umbrella": ("accessory", "umbrella"),
}


def build_record(
    image_id: int,
    category_names: list[str],
    bboxes: list[list[float]],
    image: Image.Image,
) -> dict | None:
    garments = []
    seen_slots = set()
    detected_color_slots = []

    for name, bbox in zip(category_names, bboxes):
        mapped = SLOT_MAP.get(name)
        if not mapped:
            continue
        slot, garment_type = mapped
        if slot in seen_slots:
            continue

        try:
            color = detect_color(image, bbox)
            detected_color_slots.append(slot)
        except (ValueError, OSError):
            # A degenerate (zero-area) box or unreadable region, honestly
            # null rather than a guess, exactly like scene/style below.
            # Not added to detected_color_slots since there's no color
            # here at all, confident or detected.
            color = None

        garments.append({"slot": slot, "type": garment_type, "color": color})
        seen_slots.add(slot)

    if not garments:
        return None

    caption = ", ".join(
        f"{(g['color'] + ' ') if g['color'] else ''}{g['type']} ({g['slot']})" for g in garments
    )

    return {
        "id": f"fp_{image_id}",
        "garments": garments,
        "scene": None,
        "style": None,
        "notable": [],
        "caption": caption,
        "swatch": [],
        "detected_color_slots": detected_color_slots,
    }


def main() -> None:
    dataset = load_dataset("detection-datasets/fashionpedia", split="val", streaming=True)
    category_names = dataset.features["objects"]["category"].feature.names

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for sample in dataset:
        names = [category_names[c] for c in sample["objects"]["category"]]
        image = sample["image"].convert("RGB")
        record = build_record(sample["image_id"], names, sample["objects"]["bbox"], image)
        if record is not None:
            records.append(record)
            image.save(IMAGES_DIR / f"{record['id']}.jpg", quality=90)
        if len(records) >= TARGET_COUNT:
            break

    out_path = Path(__file__).resolve().parent.parent / "app" / "data" / "real_catalog_sample.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {out_path}")
    print(f"Saved {len(records)} images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
