from signalrank.components.chunking.chunk import DocumentChunk
from signalrank.components.embeddings.base import EmbeddingProvider
from signalrank.components.retrieval.result import SearchResult
from signalrank.components.vector_store.base import VectorStore


class DenseRetriever:
    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            vector_store: VectorStore,
            chunks: dict[str, DocumentChunk],
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunks = chunks


    def retrieve(
            self,
            query: str,
            top_k: int = 10,
    ) -> list[SearchResult]:

        if not query.strip():
            return []

        query_vector = self._embedding_provider.embed_query(query)

        matches = self._vector_store.search(
            query_vector,
            top_k=top_k,
        )

        search_results: list[SearchResult] = []

        for rank, (chunk_id, score) in enumerate(
            matches,
            start=1,
        ):
            chunk = self._chunks[chunk_id]

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

        