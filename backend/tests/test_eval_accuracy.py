"""Accuracy regression test, distinct from the correctness unit tests
elsewhere: runs the 5 canonical eval queries from Working_notes.md Section 6
end to end through the API and checks the top-1 result lands in the
expected family of catalog records. Not exact-ranking-equality on purpose --
the dense half is a fallback stand-in (STRAND_DISABLE_EMBEDDINGS=1 in CI),
so this pins "did the schema-driven retrieval get the right answer", not
"did a specific tie-break resolve the same way forever".

If this test starts failing after a retrieval/scoring change, that's a real
accuracy regression worth looking at -- not a flaky test to just relax.
"""

import pytest

EVAL_QUERIES = [
    ("A bright yellow raincoat", {"img_001"}),
    ("Professional business attire inside a modern office", {"img_002", "img_011"}),
    ("Someone wearing a blue shirt sitting on a park bench", {"img_003"}),
    ("Casual weekend outfit for a city walk", {"img_004", "img_012"}),
    ("A red tie and a white shirt in a formal setting", {"img_005"}),
]


@pytest.mark.parametrize("query,expected_top1_ids", EVAL_QUERIES)
def test_eval_query_top1(client, query, expected_top1_ids):
    response = client.post("/api/query", json={"query": query, "top_k": 3})
    assert response.status_code == 200
    top1_id = response.json()["results"][0]["record"]["id"]
    assert top1_id in expected_top1_ids, f"query={query!r} got top1={top1_id!r}"


def test_compositional_query_beats_its_decoy(client):
    """The single most persuasive result from Working_notes.md Section 6
    query 5: the true match must outrank the color-swapped decoy."""
    response = client.post(
        "/api/query",
        json={"query": "A red tie and a white shirt in a formal setting", "top_k": 10},
    )
    results = {r["record"]["id"]: r["score"] for r in response.json()["results"]}
    assert results["img_005"] > results["img_006"]
