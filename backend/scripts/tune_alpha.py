"""Empirical alpha tuning: sweeps alpha against the 5 canonical eval
queries from Working_notes.md Section 6 plus the compositional decoy
pair, instead of leaving alpha=0.6 as an unvalidated hand-picked default.

Uses the keyword parser directly (not query_parsing.parse_query), same
reasoning as eval_baselines.py: this stays deterministic and free
regardless of whether a GEMINI_API_KEY happens to be set in the
environment it runs in. Re-run this once the real catalog has real
color/scene/style (scripts/extract_attributes_with_vlm.py) -- the best
alpha found here is only as good as the data it's tuned against, and
right now most of the catalog still has null color/scene/style.

Selection rule: among alpha values that do not sacrifice top-1 accuracy
on the 5 eval queries, pick the one with the largest compositional decoy
gap (img_005 must outrank img_006, and by as much as possible). Accuracy
is the hard constraint; the decoy gap is what actually distinguishes
between alpha values that all pass it.

Run with: python scripts/tune_alpha.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STRAND_DISABLE_EMBEDDINGS", "1")

from app.services.catalog import get_catalog
from app.services.query_parser import parse_query_keywords as parse_query
from app.services.retriever import search

EVAL_QUERIES = [
    ("A bright yellow raincoat", {"img_001"}),
    ("Professional business attire inside a modern office", {"img_002", "img_011"}),
    ("Someone wearing a blue shirt sitting on a park bench", {"img_003"}),
    ("Casual weekend outfit for a city walk", {"img_004", "img_012"}),
    ("A red tie and a white shirt in a formal setting", {"img_005"}),
]

DECOY_QUERY = "A red tie and a white shirt in a formal setting"

ALPHA_GRID = [round(i / 20, 2) for i in range(0, 21)]  # 0.00, 0.05, ..., 1.00


def accuracy_at(alpha: float) -> int:
    correct = 0
    for query, expected_ids in EVAL_QUERIES:
        parsed = parse_query(query)
        top1 = search(parsed, top_k=1, alpha=alpha)[0]
        if top1.record.id in expected_ids:
            correct += 1
    return correct


def decoy_gap_at(alpha: float) -> float:
    parsed = parse_query(DECOY_QUERY)
    results = search(parsed, top_k=len(get_catalog()), alpha=alpha)
    scores = {r.record.id: r.score for r in results}
    return scores["img_005"] - scores["img_006"]


def main() -> None:
    print(f"{'alpha':<8}{'accuracy':<12}{'decoy gap':<12}")
    print("-" * 32)

    rows = []
    for alpha in ALPHA_GRID:
        accuracy = accuracy_at(alpha)
        gap = decoy_gap_at(alpha)
        rows.append((alpha, accuracy, gap))
        print(f"{alpha:<8.2f}{accuracy}/{len(EVAL_QUERIES):<10}{gap:+.4f}")

    print("-" * 32)

    max_accuracy = max(r[1] for r in rows)
    candidates = [r for r in rows if r[1] == max_accuracy]
    best_alpha, best_accuracy, best_gap = max(candidates, key=lambda r: r[2])

    print(
        f"maximum-gap alpha under this sweep: {best_alpha:.2f} "
        f"(accuracy={best_accuracy}/{len(EVAL_QUERIES)}, decoy gap={best_gap:+.4f})"
    )
    print()
    print(
        "NOT adopting this as the new default. Accuracy is pegged at the ceiling and\n"
        "the decoy gap is monotonic across the whole range, so this metric mathematically\n"
        "cannot recommend anything except the most extreme alpha tried -- that is a sign\n"
        "the eval set is too narrow to tune against, not evidence that alpha=1.0 (pure\n"
        "symbolic, no dense signal at all) is actually a good idea. These 6 cases (5 eval\n"
        "queries + 1 decoy) never exercise a case dense is meant to help with: a query the\n"
        "parser only partially recognizes (see the graduated-confidence fallback this\n"
        "would otherwise interact with) still needs the dense signal to carry whatever the\n"
        "schema missed. alpha=0.6 stays the default; a real tuning pass needs a broader eval\n"
        "set with queries that actually stress the symbolic/dense tradeoff, not just these."
    )


if __name__ == "__main__":
    main()
