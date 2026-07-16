"""Part A: The Indexer, as a runnable batch workflow.

Iterates over every real Fashionpedia photo on disk and extracts a real
CLIP image embedding for each, persisting them to the on-disk vector
store at app/data/image_vector_index/ (see services/image_vector_store.py).

This is the actual Part A workflow the assignment asks for: feature
extraction from raw pixels plus efficient vector storage, using a local,
keyless model. No VLM API key is needed for this part; see
Working_notes.md Section 4.2 for what still does (color, scene, style).

Run with: pip install -r scripts/requirements-eval.txt
          python scripts/build_image_vector_index.py
Requires images already on disk (pull_fashionpedia_sample.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.image_vector_store import get_image_collection
from app.services.indexer import index_image

IMAGES_DIR = Path(__file__).resolve().parent.parent / "app" / "data" / "fashionpedia_images"


def main() -> None:
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.glob("*.jpg")):
        raise SystemExit(f"No images found at {IMAGES_DIR}. Run pull_fashionpedia_sample.py first.")

    image_paths = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Indexing {len(image_paths)} images...")

    for i, path in enumerate(image_paths, start=1):
        index_image(str(path))
        print(f"  [{i}/{len(image_paths)}] {path.stem}")

    collection = get_image_collection()
    print(f"Done. Persistent image vector collection now has {collection.count()} entries.")


if __name__ == "__main__":
    main()
