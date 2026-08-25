from fastapi.testclient import TestClient

from signalrank.api.main import app
from signalrank.components.retrieval.result import SearchResult


class FakeRetrievalService:
    def retrieve(
        self,
        query: str,
        mode: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=f"{mode}_chunk",
                doc_id="doc_test",
                text=f"Result for {query}",
                score=0.95,
                rank=1,
                source_path="test.txt",
                metadata={"mode": mode},
            )
        ]


client = TestClient(app)

TEST_SERVICE_TOKEN = "test-service-token"

app.state.retrieval_service = FakeRetrievalService()
app.state.service_token = TEST_SERVICE_TOKEN

AUTH_HEADERS = {
    "X-SignalRank-Service-Token": TEST_SERVICE_TOKEN,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_dense_retrieval_endpoint():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "robotic vehicle exploring the red planet",
            "mode": "dense",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == ("robotic vehicle exploring the red planet")
    assert body["mode"] == "dense"

    assert len(body["results"]) == 1
    assert body["results"][0]["chunk_id"] == "dense_chunk"
    assert body["results"][0]["score"] == 0.95
    assert body["results"][0]["rank"] == 1


def test_bm25_retrieval_endpoint():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "Mars rover samples",
            "mode": "bm25",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["mode"] == "bm25"
    assert body["results"][0]["chunk_id"] == "bm25_chunk"


def test_hybrid_retrieval_endpoint():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "Mars rover samples",
            "mode": "hybrid",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["mode"] == "hybrid"
    assert body["results"][0]["chunk_id"] == "hybrid_chunk"


def test_invalid_retrieval_mode_is_rejected():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "test query",
            "mode": "gulugulu",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_top_k_must_be_positive():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "test query",
            "mode": "dense",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_retrieve_rejects_missing_service_token():
    response = client.post(
        "/retrieve",
        json={
            "query": "Mars rover",
            "mode": "dense",
            "top_k": 3,
        },
    )

    assert response.status_code == 401


def test_retrieve_rejects_invalid_service_token():
    response = client.post(
        "/retrieve",
        headers={
            "X-SignalRank-Service-Token": "wrong-token",
        },
        json={
            "query": "Mars rover",
            "mode": "dense",
            "top_k": 3,
        },
    )

    assert response.status_code == 401


def test_top_k_above_limit_is_rejected():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "Mars rover",
            "mode": "dense",
            "top_k": 11,
        },
    )

    assert response.status_code == 422


def test_query_above_length_limit_is_rejected():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "x" * 501,
            "mode": "dense",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_blank_query_is_rejected():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "  ",
            "mode": "dense",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_top_k_at_limit_is_allowed():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "Mars rover",
            "mode": "dense",
            "top_k": 10,
        },
    )

    assert response.status_code == 200


def test_query_at_length_limit_is_allowed():
    response = client.post(
        "/retrieve",
        headers=AUTH_HEADERS,
        json={
            "query": "x" * 500,
            "mode": "dense",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
