import pytest

from signalrank.components.retrieval.result import SearchResult
from signalrank.services.ranking_service import RankingService


class FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        return list(reversed(candidates))[:top_k]


def make_result(
    chunk_id: str,
    rank: int,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc_{chunk_id}",
        text=f"text {chunk_id}",
        score=1.0,
        rank=rank,
        source_path="test.txt",
        metadata={},
    )


def test_ranking_service_uses_selected_reranker() -> None:
    candidates = [
        make_result("a", 1),
        make_result("b", 2),
        make_result("c", 3),
    ]

    service = RankingService(
        rerankers={
            "flashrank": FakeReranker(),
        }
    )

    results = service.rerank(
        query="test query",
        candidates=candidates,
        mode="flashrank",
        top_k=2,
    )

    assert [result.chunk_id for result in results] == [
        "c",
        "b",
    ]


def test_ranking_service_rejects_unknown_mode() -> None:
    service = RankingService(
        rerankers={
            "flashrank": FakeReranker(),
        }
    )

    with pytest.raises(
        ValueError,
        match="Unknown ranking mode",
    ):
        service.rerank(
            query="test",
            candidates=[],
            mode="unknown",
        )
