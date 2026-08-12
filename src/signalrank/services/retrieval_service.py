from signalrank.components.retrieval.base import Retriever
from signalrank.components.retrieval.result import SearchResult


class RetrievalService:
    def __init__(
            self,
            retrievers: dict[str, Retriever],
    ):
        self._retrievers = retrievers
   
    def retrieve(
            self,
            query: str,
            mode: str,
            top_k: int = 10,
    ) -> list[SearchResult]:
        try:
            retriever = self._retrievers[mode]
        except KeyError as exc:
            raise ValueError(
                f"Unknown retrieval mode: {mode}"
            ) from exc
        
        return retriever.retrieve(
            query,
            top_k,
        )