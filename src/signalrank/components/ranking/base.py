from typing import Protocol

from signalrank.components.retrieval.result import SearchResult


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        *,
        top_k: int = 10,
    ) -> list[SearchResult]: ...
