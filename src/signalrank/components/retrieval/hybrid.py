from collections import defaultdict

from signalrank.components.retrieval.base import Retriever
from signalrank.components.retrieval.result import SearchResult


class HybridRetriever:
    """
    Fuse lexical and dense retrieval results using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")

        self._bm25_retriever = bm25_retriever
        self._dense_retriever = dense_retriever
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        candidate_k = top_k * self._candidate_multiplier

        retrieval_results = {
            "bm25": self._bm25_retriever.retrieve(
                query,
                top_k=candidate_k,
            ),
            "dense": self._dense_retriever.retrieve(
                query,
                top_k=candidate_k,
            ),
        }

        fused_scores: dict[str, float] = defaultdict(float)
        source_results: dict[str, SearchResult] = {}
        fusion_metadata: dict[str, dict[str, object]] = defaultdict(dict)

        # 1. Accumulate evidence from BOTH retrievers.

        for source_name, results in retrieval_results.items():
            for result in results:
                chunk_id = result.chunk_id

                source_results.setdefault(
                    chunk_id,
                    result,
                )

                fused_scores[chunk_id] += 1.0 / (self._rrf_k + result.rank)

                fusion_metadata[chunk_id][f"{source_name}_rank"] = result.rank

                fusion_metadata[chunk_id][f"{source_name}_score"] = result.score

        # 2. Rank the fused scores.
        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                chunk_id,
            ),
        )[:top_k]

        fused_results: list[SearchResult] = []

        for rank, chunk_id in enumerate(
            ranked_chunk_ids,
            start=1,
        ):
            source_result = source_results[chunk_id]

            fused_results.append(
                SearchResult(
                    chunk_id=source_result.chunk_id,
                    doc_id=source_result.doc_id,
                    text=source_result.text,
                    score=fused_scores[chunk_id],
                    rank=rank,
                    source_path=source_result.source_path,
                    metadata={
                        **source_result.metadata,
                        "retrieval_mode": "hybrid",
                        "fusion_method": "rrf",
                        **fusion_metadata[chunk_id],
                    },
                )
            )

        return fused_results
