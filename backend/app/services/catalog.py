"""Loads the combined catalog: hand-written mock decoy pairs plus the
real Fashionpedia sample, both in app/data/."""

import json
from importlib import resources

from app.schema import ImageRecord

CATALOG_FILES = ["sample_catalog.json", "real_catalog_sample.json"]


def _load_records(filename: str) -> list[ImageRecord]:
    data_path = resources.files("app.data").joinpath(filename)
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    return [ImageRecord(**item) for item in raw]


def load_catalog() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for filename in CATALOG_FILES:
        records.extend(_load_records(filename))
    return records


_CATALOG: list[ImageRecord] | None = None


def get_catalog() -> list[ImageRecord]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_catalog()
    return _CATALOG
