import pytest

from signalrank.components.retrieval.result import SearchResult
from signalrank.services.retrieval_service import RetrievalService


class FakeRetriever:
    def __init__(
        self,
        chunk_id: str,
    ):
        self._chunk_id = chunk_id

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=self._chunk_id,
                doc_id="doc_test",
                text=query,
                score=1.0,
                rank=1,
                source_path="text.txt",
            )
        ]


def test_retrieval_service_routes_to_selected_retriever():
    service = RetrievalService(
        retrievers={
            "bm25": FakeRetriever("bm25_chunk"),
            "dense": FakeRetriever("dense_chunk"),
        }
    )

    results = service.retrieve(
        query="test_query",
        mode="dense",
    )

    assert results[0].chunk_id == "dense_chunk"


def test_retrieval_service_rejects_unknown_mode():
    service = RetrievalService(
        retrievers={
            "bm25": FakeRetriever("bm25_chunk"),
        }
    )

    with pytest.raises(
        ValueError,
        match="Unknown retrieval mode: gulugulu",
    ):
        service.retrieve(
            query="test query",
            mode="gulugulu",
        )
