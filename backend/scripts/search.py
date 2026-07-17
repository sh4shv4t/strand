"""Plain CLI entrypoint: "a script that accepts a natural language string
and returns the top k matching images", the assignment's literal Part B
requirement. The FastAPI + React app already does this and more; this is
a zero-setup, one-line way to run the same retrieval logic without
starting a server or a browser.

Uses the exact same code path the running app uses (query_parsing.parse_query
+ retriever.search), not a separate reimplementation, so this script's
behavior never drifts from the API's: it automatically uses the real LLM
query parser once GEMINI_API_KEY is set, and the same keyword-spotting
fallback otherwise.

Run with: python scripts/search.py "a red tie and a white shirt in a formal setting"
Optional: --top-k (default 5), --alpha (default 0.6, see Working_notes.md
Section 3 for what alpha trades off).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.query_parsing import parse_query
from app.services.retriever import search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help='Natural language description, e.g. "a bright yellow raincoat"')
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()

    parsed = parse_query(args.query)
    results = search(parsed, top_k=args.top_k, alpha=args.alpha)

    garment_summary = [f"{g.color + ' ' if g.color else ''}{g.type}".strip() for g in parsed.garments]
    print(f'Query: "{args.query}"')
    print(
        f"Parsed -> garments={garment_summary} scene={parsed.scene} "
        f"style={parsed.style} confidence={parsed.confidence:.2f}\n"
    )

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.record.id}  score={result.score:.4f}  "
            f"(symbolic={result.symbolic_score:.4f}, dense={result.dense_score:.4f})"
        )
        print(f"   caption: {result.record.caption}")
        if result.matched_fields:
            print(f"   matched: {', '.join(result.matched_fields)}")
        print()


if __name__ == "__main__":
    main()
