from signalrank.components.vector_store.qdrant import QdrantVectorStore


def test_qdrant_returns_nearest_vector():
    store = QdrantVectorStore(
        dimension=3,
        collection_name="test_vectors",
    )

    store.upsert(
        ids=[
            "chunk_1",
            "chunk_2",
            "chunk_3",
        ],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        payloads=[
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
        ],
    )

    results = store.search(
        query_vector=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2

    assert results[0][0] == "chunk_1"
    assert results[0][1] > results[1][1]