from signalrank.components.retrieval.hybrid import HybridRetriever
from signalrank.components.retrieval.result import SearchResult


class FakeRetriever:
    def __init__(
        self,
        results: list[SearchResult],
    ):
        self._results = results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        return self._results[:top_k]


def make_result(
    chunk_id: str,
    score: float,
    rank: int,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="doc_test",
        text=f"Text for {chunk_id}",
        score=score,
        rank=rank,
        source_path="test.txt",
    )


def test_hybrid_retriever_combines_ranking():
    bm25 = FakeRetriever(
        [
            make_result("chunk_a", 12.0, 1),
            make_result("chunk_b", 10.0, 2),
            make_result("chunk_c", 8.0, 3),
        ]
    )

    dense = FakeRetriever(
        [
            make_result("chunk_b", 0.95, 1),
            make_result("chunk_d", 0.90, 2),
            make_result("chunk_a", 0.85, 3),
        ]
    )

    retriever = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        rrf_k=60,
        candidate_multiplier=4,
    )

    results = retriever.retrieve(
        "test_query",
        top_k=3,
    )

    assert [result.chunk_id for result in results] == [
        "chunk_b",
        "chunk_a",
        "chunk_d",
    ]

    assert results[0].rank == 1
    assert results[0].metadata["bm25_rank"] == 2
    assert results[0].metadata["dense_rank"] == 1
