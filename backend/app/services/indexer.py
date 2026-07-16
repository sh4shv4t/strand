"""Stub for the offline indexing loop described in Working_notes.md Section 4.1.

Not implemented: this scaffold ships a hand-written mock catalog
(`app/data/sample_catalog.json`) instead of running a real indexing pass.
This function documents the steps a real implementation would perform,
so the eventual integration has a clear seam to land in.
"""

from app.schema import ImageRecord


def index_image(image_path: str) -> ImageRecord:
    """Would run, per Working_notes.md Section 4.1:

    1. VLM/attribute extraction (Option D: one VLM call -> JSON schema).
    2. Flattened-caption construction + dense embedding pass (fashion-tuned encoder).
    3. Scene/style zero-shot tagging where the dataset doesn't label it natively.
    4. Write the resulting ImageRecord + embedding to the vector store.

    Raises NotImplementedError -- this scaffold only serves the mock catalog.
    """
    raise NotImplementedError(
        "Real indexing is not implemented in this scaffold. "
        "See Working_notes.md Section 4.1 for the intended pipeline; "
        f"received image_path={image_path!r} with nothing to do with it yet."
    )
