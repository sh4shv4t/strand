def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_ranked_results(client):
    response = client.post(
        "/api/query",
        json={"query": "a red tie and a white shirt in a formal setting", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 3
    assert body["results"][0]["record"]["id"] == "img_005"
    assert body["parsed"]["style"] == "formal"


def test_query_endpoint_validates_request(client):
    response = client.post("/api/query", json={})
    assert response.status_code == 422


def test_catalog_endpoint_returns_full_catalog(client):
    response = client.get("/api/catalog")
    assert response.status_code == 200
    body = response.json()
    assert len(body) > 0
    assert all("id" in record and "caption" in record for record in body)


def test_index_endpoint_is_not_implemented(client):
    response = client.post("/api/index", params={"image_path": "foo.jpg"})
    assert response.status_code == 501


def test_unknown_route_returns_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
