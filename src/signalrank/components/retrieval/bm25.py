import bm25s

from signalrank.components.chunking.chunk import DocumentChunk
from signalrank.components.retrieval.result import SearchResult


class BM25Retriever:
    def __init__(
            self,
            chunks: list[DocumentChunk],
    ):
        self._chunks = chunks

        corpus = [chunk.text for chunk in chunks]

        corpus_tokens = bm25s.tokenize(corpus)

        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)

    def retrieve(
            self,
            query: str,
            top_k: int = 10,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        if not self._chunks:
            return []

        query_tokens = bm25s.tokenize(query)

        result_indices, scores = self._retriever.retrieve(
            query_tokens,
            k=min(top_k, len(self._chunks)),
        )

        search_results: list[SearchResult] = []

        for rank, (chunk_index, score) in enumerate(
            zip(result_indices[0], scores[0]),
            start=1,
        ):
            chunk = self._chunks[int(chunk_index)]

            search_results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(score),
                    rank=rank,
                    source_path=chunk.source_path,
                    metadata=dict(chunk.metadata),
                )
            )

        return search_results