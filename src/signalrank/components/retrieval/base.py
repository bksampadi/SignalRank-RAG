from typing import Protocol

from signalrank.components.retrieval.result import SearchResult


class Retriever(Protocol):
    def retriever(
            self,
            query: str,
            top_k: int = 10,
    ) -> list[SearchResult]:
        ...