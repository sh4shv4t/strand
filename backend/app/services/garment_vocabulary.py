"""Canonical garment-type normalization for symbolic matching.

retriever.py's _garment_matches compares Garment.type by exact string
equality, but type strings come from independently-worded vocabularies:
the keyword query parser (query_parser.py's GARMENT_SLOTS), Fashionpedia's
own ground-truth categories (pull_fashionpedia_sample.py's SLOT_MAP), and
eventually an open-vocabulary VLM. Several of these are the literal same
garment under a different spelling, and exact equality silently fails
them all: a query for "shoes" (the parser's plural form) never matched
any real catalog record, all of which are typed "shoe" (Fashionpedia's
singular ground-truth category), and "blouse" never matched a record
typed "shirt" even though Fashionpedia's own taxonomy groups them as one
category ("shirt, blouse", see SLOT_MAP). canonical_type() collapses
known synonym pairs to one representative string before comparison so
these actually match.

This is a deliberately conservative, curated list, not general fuzzy /
edit-distance string matching. That was tried and rejected: computing
difflib.SequenceMatcher ratios across this vocabulary found "shirt" vs
"skirt" at 0.80 similarity, higher than several genuine plural pairs
("jean"/"jeans" at 0.889, "shoe"/"shoes" at 0.889), and "jacket" vs the
unrelated word "racket" at 0.833. Any single similarity threshold either
merges shirt/skirt (a real, wrong match) or is too strict to catch most
real typos at all, short garment words are too dense in edit-distance
space for a threshold to separate synonyms from unrelated words. See
test_garment_vocabulary.py for the measured ratios.

Garments that merely share a slot but are stylistically distinct (jacket
vs coat vs blazer, sneakers vs heels vs boots) are also deliberately NOT
merged, even though they're semantically related: Fashionpedia's ground
truth has no footwear-style granularity at all (every real catalog
footwear record is typed "shoe" regardless of actual style), so merging
"sneakers" into "shoe" would fabricate a specific visual claim the data
doesn't support, a query for sneakers would then "match" a record that's
actually wearing heels. That's a data-granularity gap a real VLM call
would need to close, not something a smarter string comparison can fix
honestly.
"""

_SYNONYMS: dict[str, str] = {
    # Fashionpedia's own ground truth groups these under one category
    # (SLOT_MAP: "shirt, blouse" and "top, t-shirt, sweatshirt"); the
    # keyword parser gives them separate type strings.
    "blouse": "shirt",
    "top": "t-shirt",
    "sweatshirt": "t-shirt",
    # pure spelling variants of the identical word.
    "tshirt": "t-shirt",
    "tee": "t-shirt",
    # singular/plural, the exact bug that motivated this module: a query
    # for "shoes" never matched the real catalog's singular "shoe" type.
    "shoes": "shoe",
    "gloves": "glove",
    # regional synonym for the identical garment (US vs UK naming).
    "trousers": "pants",
}


def canonical_type(raw_type: str) -> str:
    """Maps a garment type string to one canonical spelling. Unknown
    types are returned lowercased/stripped, unchanged otherwise."""
    normalized = raw_type.strip().lower()
    return _SYNONYMS.get(normalized, normalized)
