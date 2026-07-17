def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["catalog_size"] > 0
    assert isinstance(body["using_real_embeddings"], bool)


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


def test_index_endpoint_success(client, monkeypatch):
    monkeypatch.setattr("app.routers.index.index_image", lambda path: [0.1, 0.2, 0.3])
    response = client.post("/api/index", json={"image_path": "foo.jpg"})
    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "embedding_dim": 3, "registered": False}


def test_index_endpoint_reports_missing_dependencies(client, monkeypatch):
    from app.services.indexer import IndexingDependenciesMissing

    def raise_missing(path):
        raise IndexingDependenciesMissing("deps not installed")

    monkeypatch.setattr("app.routers.index.index_image", raise_missing)
    response = client.post("/api/index", json={"image_path": "foo.jpg"})
    assert response.status_code == 503


def test_index_endpoint_reports_missing_file(client, monkeypatch):
    def raise_not_found(path):
        raise FileNotFoundError(f"no such file: {path}")

    monkeypatch.setattr("app.routers.index.index_image", raise_not_found)
    response = client.post("/api/index", json={"image_path": "does-not-exist.jpg"})
    assert response.status_code == 404


def test_index_endpoint_registers_a_searchable_record_when_garments_supplied(client, monkeypatch):
    # retriever._CATALOG and catalog._CATALOG are the same list object in
    # production (register_record relies on that sharing, see
    # retriever.py), so both must be repointed to the same isolated copy
    # here, otherwise this test would either miss its own mutation (if
    # only one were patched) or permanently pollute every later test with
    # a "new_arrival" record (if neither were).
    import app.services.catalog as catalog_module
    import app.services.retriever as retriever_module

    isolated_catalog = list(catalog_module.get_catalog())
    monkeypatch.setattr(catalog_module, "_CATALOG", isolated_catalog)
    monkeypatch.setattr(retriever_module, "_CATALOG", isolated_catalog)

    monkeypatch.setattr("app.routers.index.index_image", lambda path: [0.1, 0.2, 0.3])
    response = client.post(
        "/api/index",
        json={
            "image_path": "new_arrival.jpg",
            "garments": [{"slot": "upper", "type": "shirt", "color": "green"}],
            "scene": "office",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "embedding_dim": 3, "registered": True}

    catalog_response = client.get("/api/catalog")
    ids = [record["id"] for record in catalog_response.json()]
    assert "new_arrival" in ids

    query_response = client.post(
        "/api/query", json={"query": "a green shirt in an office", "top_k": 1}
    )
    assert query_response.json()["results"][0]["record"]["id"] == "new_arrival"


def test_unknown_route_returns_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
