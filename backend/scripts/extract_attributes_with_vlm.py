"""Fills in color, scene, and style on the real Fashionpedia records using
a real VLM call (Gemini), replacing the honest nulls left by
pull_fashionpedia_sample.py.

Garment presence (slot + type) stays as Fashionpedia's own ground truth,
already reliable and already real (see indexer.py's docstring for why
that is a legitimate form of feature extraction on its own); this script
only asks the VLM for what ground truth does not carry: color per
garment, plus scene and style for the whole image. The VLM's own
garments are matched back to the existing ones by slot, purely to read
off a color; garments it sees that are not in the ground-truth list are
ignored, garment presence is not something to take a VLM's word over a
labeled dataset for.

Run with: pip install -r scripts/requirements-eval.txt --no-deps google-genai
          (or just pip install google-genai; it is a light dependency,
          already in backend/requirements.txt)
          python scripts/extract_attributes_with_vlm.py
Requires GEMINI_API_KEY (see backend/.env.example) and images already on
disk (pull_fashionpedia_sample.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.gemini_client import GeminiNotConfigured
from app.services.vlm_attribute_extractor import extract_attributes

BACKEND_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BACKEND_DIR / "app" / "data" / "fashionpedia_images"
CATALOG_PATH = BACKEND_DIR / "app" / "data" / "real_catalog_sample.json"


def build_caption(record: dict) -> str:
    garment_bits = [f"{g['color'] + ' ' if g.get('color') else ''}{g['type']} ({g['slot']})" for g in record["garments"]]
    parts = list(garment_bits)
    if record.get("style"):
        parts.append(f"{record['style']} style")
    if record.get("scene"):
        parts.append(f"{record['scene']} setting")
    return ", ".join(parts)


def main() -> None:
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.glob("*.jpg")):
        raise SystemExit(f"No images found at {IMAGES_DIR}. Run pull_fashionpedia_sample.py first.")

    records = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    records_by_id = {r["id"]: r for r in records}

    updated = 0
    for path in sorted(IMAGES_DIR.glob("*.jpg")):
        record = records_by_id.get(path.stem)
        if record is None:
            continue

        try:
            attributes = extract_attributes(str(path))
        except GeminiNotConfigured as exc:
            raise SystemExit(str(exc))

        colors_by_slot = {g.slot: g.color for g in attributes.garments if g.color}
        for garment in record["garments"]:
            if garment["slot"] in colors_by_slot:
                garment["color"] = colors_by_slot[garment["slot"]]

        record["scene"] = attributes.scene
        record["style"] = attributes.style
        record["caption"] = build_caption(record)

        updated += 1
        print(f"  [{updated}] {record['id']}: scene={attributes.scene} style={attributes.style}")

    CATALOG_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Updated {updated} records in {CATALOG_PATH}")


if __name__ == "__main__":
    main()
