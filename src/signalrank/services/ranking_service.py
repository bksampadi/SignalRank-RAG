from signalrank.components.ranking.base import Reranker
from signalrank.components.retrieval.result import SearchResult


class RankingService:
    def __init__(self, rerankers: dict[str, Reranker]):
        self._rerankers = rerankers

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        *,
        mode: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        try:
            reranker = self._rerankers[mode]
        except KeyError as exc:
            raise ValueError(f"Unknown ranking mode: {mode}") from exc

        return reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )
