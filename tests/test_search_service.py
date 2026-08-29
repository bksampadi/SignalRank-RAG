from unittest.mock import MagicMock

from signalrank.services.ranking_service import RankingService
from signalrank.services.retrieval_service import RetrievalService
from signalrank.services.search_service import SearchService


def test_search_expands_candidates_before_reranking() -> None:
    retrieval_service = MagicMock(
        spec=RetrievalService,
    )
    ranking_service = MagicMock(
        spec=RankingService,
    )

    candidates = [MagicMock() for _ in range(12)]
    ranked = candidates[:3]

    retrieval_service.retrieve.return_value = candidates
    ranking_service.rerank.return_value = ranked

    service = SearchService(
        retrieval_service=retrieval_service,
        ranking_service=ranking_service,
        ranking_mode="flashrank",
        ranking_enabled=True,
        candidate_multiplier=4,
    )

    results = service.search(
        query="dinosaur extinction",
        retrieval_mode="hybrid",
        top_k=3,
    )

    retrieval_service.retrieve.assert_called_once_with(
        query="dinosaur extinction",
        mode="hybrid",
        top_k=12,
    )

    ranking_service.rerank.assert_called_once_with(
        query="dinosaur extinction",
        candidates=candidates,
        mode="flashrank",
        top_k=3,
    )

    assert results == ranked
