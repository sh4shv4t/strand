"""One-off script: pull a small real sample from Fashionpedia and write it
to app/data/real_catalog_sample.json in the Strand schema. Also persists
the actual JPEGs to app/data/fashionpedia_images/ (gitignored -- rerun this
script to regenerate them; they're not committed) for the Tier 2 real-image
CLIP baseline in scripts/eval_clip_baseline.py.

Not part of the running app -- run manually to (re)generate the sample:

    pip install datasets pillow
    python scripts/pull_fashionpedia_sample.py

Uses only the dataset's own ground-truth category labels (bbox category,
mapped to our garment slot/type taxonomy). This HF mirror of Fashionpedia
(`detection-datasets/fashionpedia`) exposes bbox + category only, not the
294 fine-grained attributes from the original paper, so there is no real
color signal here -- color stays null rather than guessed, and scene/style
stay null too (Working_notes.md Section 4.1 step 4: environment/style
tagging is a separate zero-shot/VLM step, deliberately deferred until a
VLM API key or local model is wired up). This script only replaces the
mocked *garment detection* half of indexing with real data.

Streaming iteration order is deterministic (no shuffling), so rerunning
this reproduces the same image_ids as the committed real_catalog_sample.json.

TARGET_COUNT is 1000 (the top of the assignment's stated 500-1,000 image
range): an earlier pass ran this at 40 for fast local iteration during
scaffolding, and that smaller sample was never re-pulled at the size the
assignment actually asks for before this.
"""

import json
from pathlib import Path

from datasets import load_dataset

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


def build_record(image_id: int, category_names: list[str]) -> dict | None:
    garments = []
    seen_slots = set()

    for name in category_names:
        mapped = SLOT_MAP.get(name)
        if not mapped:
            continue
        slot, garment_type = mapped
        if slot in seen_slots:
            continue
        garments.append({"slot": slot, "type": garment_type, "color": None})
        seen_slots.add(slot)

    if not garments:
        return None

    caption = ", ".join(f"{g['type']} ({g['slot']})" for g in garments)

    return {
        "id": f"fp_{image_id}",
        "garments": garments,
        "scene": None,
        "style": None,
        "notable": [],
        "caption": caption,
        "swatch": [],
    }


def main() -> None:
    dataset = load_dataset("detection-datasets/fashionpedia", split="val", streaming=True)
    category_names = dataset.features["objects"]["category"].feature.names

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for sample in dataset:
        names = [category_names[c] for c in sample["objects"]["category"]]
        record = build_record(sample["image_id"], names)
        if record is not None:
            records.append(record)
            sample["image"].convert("RGB").save(IMAGES_DIR / f"{record['id']}.jpg", quality=90)
        if len(records) >= TARGET_COUNT:
            break

    out_path = Path(__file__).resolve().parent.parent / "app" / "data" / "real_catalog_sample.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {out_path}")
    print(f"Saved {len(records)} images to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
