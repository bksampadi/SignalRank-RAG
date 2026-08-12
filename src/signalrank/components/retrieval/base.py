from typing import Protocol

from signalrank.components.retrieval.result import SearchResult

class Retriever(Protocol):
    def retrieve(
            self,
            query: str,
            top_k: int = 10,
    ) -> list[SearchResult]:
        ...