"""Baseline comparison: dense-only retrieval (alpha=0, a stand-in for
Option A/B's "single pooled embedding" from Working_notes.md Section 2)
vs. our weighted-hybrid system (alpha=0.6), on the 5 canonical eval
queries from Section 6 plus the compositional decoy pair.

Run with: python scripts/eval_baselines.py
Prints a comparison table -- the numbers this prints are what
Working_notes.md Section 12 cites.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STRAND_DISABLE_EMBEDDINGS", "1")

from app.services.catalog import get_catalog
from app.services.query_parser import parse_query
from app.services.retriever import search

EVAL_QUERIES = [
    ("A bright yellow raincoat", {"img_001"}),
    ("Professional business attire inside a modern office", {"img_002", "img_011"}),
    ("Someone wearing a blue shirt sitting on a park bench", {"img_003"}),
    ("Casual weekend outfit for a city walk", {"img_004", "img_012"}),
    ("A red tie and a white shirt in a formal setting", {"img_005"}),
]


def top1(query: str, alpha: float) -> tuple[str, float]:
    parsed = parse_query(query)
    results = search(parsed, top_k=1, alpha=alpha)
    return results[0].record.id, results[0].score


def decoy_scores(alpha: float) -> tuple[float, float]:
    parsed = parse_query("A red tie and a white shirt in a formal setting")
    results = search(parsed, top_k=len(get_catalog()), alpha=alpha)
    scores = {r.record.id: r.score for r in results}
    return scores["img_005"], scores["img_006"]


def main() -> None:
    print(f"{'query':<55} {'dense-only top1':<18} {'hybrid top1':<18} expected")
    print("-" * 110)

    dense_correct = 0
    hybrid_correct = 0

    for query, expected in EVAL_QUERIES:
        dense_id, _ = top1(query, alpha=0.0)
        hybrid_id, _ = top1(query, alpha=0.6)
        dense_ok = dense_id in expected
        hybrid_ok = hybrid_id in expected
        dense_correct += dense_ok
        hybrid_correct += hybrid_ok
        print(
            f"{query:<55} {dense_id + (' ok' if dense_ok else ' X'):<18} "
            f"{hybrid_id + (' ok' if hybrid_ok else ' X'):<18} {sorted(expected)}"
        )

    print("-" * 110)
    print(f"dense-only top1 accuracy:  {dense_correct}/{len(EVAL_QUERIES)}")
    print(f"hybrid top1 accuracy:      {hybrid_correct}/{len(EVAL_QUERIES)}")

    print()
    print("Compositional decoy test (img_005 = true match, img_006 = color-swapped decoy):")
    dense_005, dense_006 = decoy_scores(alpha=0.0)
    hybrid_005, hybrid_006 = decoy_scores(alpha=0.6)
    print(f"  dense-only:  img_005={dense_005:.4f}  img_006={dense_006:.4f}  gap={dense_005 - dense_006:+.4f}")
    print(f"  hybrid:      img_005={hybrid_005:.4f}  img_006={hybrid_006:.4f}  gap={hybrid_005 - hybrid_006:+.4f}")


if __name__ == "__main__":
    main()
