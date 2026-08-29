from signalrank.components.retrieval.result import SearchResult
from signalrank.services.ranking_service import RankingService
from signalrank.services.retrieval_service import RetrievalService


class SearchService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        ranking_service: RankingService,
        *,
        ranking_mode: str = "flashrank",
        ranking_enabled: bool = True,
        candidate_multiplier: int = 4,
    ):
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")

        self._retrieval_service = retrieval_service
        self._ranking_service = ranking_service
        self._ranking_mode = ranking_mode
        self._ranking_enabled = ranking_enabled
        self._candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        retrieval_mode: str,
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if not self._ranking_enabled:
            return self._retrieval_service.retrieve(
                query=query,
                mode=retrieval_mode,
                top_k=top_k,
            )

        candidate_k = top_k * self._candidate_multiplier

        candidates = self._retrieval_service.retrieve(
            query=query,
            mode=retrieval_mode,
            top_k=candidate_k,
        )

        return self._ranking_service.rerank(
            query=query,
            candidates=candidates,
            mode=self._ranking_mode,
            top_k=top_k,
        )
