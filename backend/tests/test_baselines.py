"""Tier 1 baseline comparison: dense-only retrieval (alpha=0, a stand-in
for Option A/B's single pooled embedding from Working_notes.md Section 2)
vs. our weighted-hybrid system (alpha=0.6). See scripts/eval_baselines.py
for the full printed comparison this pins numbers from.
"""

from app.schema import ParsedQuery
from app.services.retriever import get_catalog, search


def _decoy_scores(alpha: float) -> tuple[float, float]:
    parsed = ParsedQuery(
        raw_query="a red tie and a white shirt in a formal setting",
        garments=[
            {"slot": "upper", "type": "shirt", "color": "white"},
            {"slot": "accessory", "type": "tie", "color": "red"},
        ],
        style="formal",
    )
    results = search(parsed, top_k=len(get_catalog()), alpha=alpha)
    scores = {r.record.id: r.score for r in results}
    return scores["img_005"], scores["img_006"]


def test_dense_only_cannot_separate_the_compositional_decoy():
    """The whole point of this project: a single pooled embedding (dense
    similarity alone) treats "red tie, white shirt" and "white tie, red
    shirt" as equally good matches, because bag-of-words captions are
    identical regardless of which word attaches to which garment."""
    true_match, decoy = _decoy_scores(alpha=0.0)
    assert true_match == decoy


def test_hybrid_separates_the_compositional_decoy():
    true_match, decoy = _decoy_scores(alpha=0.6)
    assert true_match > decoy


def test_hybrid_gap_exceeds_dense_only_gap():
    dense_true, dense_decoy = _decoy_scores(alpha=0.0)
    hybrid_true, hybrid_decoy = _decoy_scores(alpha=0.6)
    dense_gap = dense_true - dense_decoy
    hybrid_gap = hybrid_true - hybrid_decoy
    assert hybrid_gap > dense_gap
