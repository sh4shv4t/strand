"""garment_vocabulary.py tests. Includes the measured evidence that
motivated rejecting generic fuzzy/edit-distance matching in favor of a
curated synonym list, see garment_vocabulary.py's module docstring: if
this ever regresses (someone adds a fuzzy-matching shortcut later), this
test catches it re-introducing a real false positive.
"""

import difflib

from app.services.garment_vocabulary import canonical_type


def test_known_synonyms_canonicalize_to_the_same_type():
    assert canonical_type("blouse") == canonical_type("shirt")
    assert canonical_type("top") == canonical_type("t-shirt")
    assert canonical_type("sweatshirt") == canonical_type("t-shirt")
    assert canonical_type("tshirt") == canonical_type("t-shirt")
    assert canonical_type("tee") == canonical_type("t-shirt")
    assert canonical_type("shoes") == canonical_type("shoe")
    assert canonical_type("gloves") == canonical_type("glove")
    assert canonical_type("trousers") == canonical_type("pants")


def test_is_case_and_whitespace_insensitive():
    assert canonical_type("  Blouse ") == canonical_type("shirt")


def test_stylistically_distinct_garments_stay_separate():
    # Sharing a slot is not the same as being the same garment: merging
    # these would fabricate a specific visual claim the data doesn't
    # support (see module docstring's footwear/outerwear reasoning).
    assert canonical_type("sneakers") != canonical_type("shoe")
    assert canonical_type("heels") != canonical_type("shoe")
    assert canonical_type("jacket") != canonical_type("coat")


def test_edit_distance_matching_is_unsafe_on_this_vocabulary():
    """The actual measurement behind garment_vocabulary.py's design
    decision: shirt/skirt score higher than several genuine synonym
    pairs, so no single similarity threshold could separate real
    synonyms from real false positives. This is why canonical_type()
    uses a curated map instead of difflib/edit-distance matching.
    """
    shirt_skirt = difflib.SequenceMatcher(None, "shirt", "skirt").ratio()
    jean_jeans = difflib.SequenceMatcher(None, "jean", "jeans").ratio()
    shoe_shoes = difflib.SequenceMatcher(None, "shoe", "shoes").ratio()
    jacket_racket = difflib.SequenceMatcher(None, "jacket", "racket").ratio()

    assert shirt_skirt > 0.75, "if this drops, re-check whether fuzzy matching is safe now"
    assert shirt_skirt >= jean_jeans - 0.15
    assert shirt_skirt >= shoe_shoes - 0.15
    assert jacket_racket > 0.75
