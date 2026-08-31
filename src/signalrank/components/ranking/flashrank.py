import os
from typing import TypedDict, cast

from flashrank import Ranker, RerankRequest

from signalrank.components.retrieval.result import SearchResult


class _FlashRankResult(TypedDict):
    id: str
    text: str
    score: float


class FlashRankReranker:
    """
    Rerank retrieved candidates using a FlashRank cross-encoder.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        self._model_name = model_name

        cache_dir = os.getenv(
            "FLASHRANK_CACHE_DIR",
            "/tmp",
        )

        self._ranker = Ranker(
            model_name=model_name,
            cache_dir=cache_dir,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not candidates:
            return []

        passages = [
            {
                "id": candidate.chunk_id,
                "text": candidate.text,
            }
            for candidate in candidates
        ]

        request = RerankRequest(
            query=query,
            passages=passages,
        )

        ranked = cast(
            list[_FlashRankResult],
            self._ranker.rerank(request),
        )

        candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}

        reranked_results: list[SearchResult] = []

        for rank, result in enumerate(
            ranked[:top_k],
            start=1,
        ):
            source = candidates_by_id[result["id"]]
            reranker_score = float(result["score"])

            metadata = dict(source.metadata)

            metadata.setdefault(
                "retrieval_score",
                source.score,
            )
            metadata.setdefault(
                "retrieval_rank",
                source.rank,
            )

            metadata.update(
                {
                    "reranker": "flashrank",
                    "reranker_model": self._model_name,
                    "reranker_score": reranker_score,
                }
            )

            reranked_results.append(
                SearchResult(
                    chunk_id=source.chunk_id,
                    doc_id=source.doc_id,
                    text=source.text,
                    score=reranker_score,
                    rank=rank,
                    source_path=source.source_path,
                    metadata=metadata,
                )
            )

        return reranked_results
