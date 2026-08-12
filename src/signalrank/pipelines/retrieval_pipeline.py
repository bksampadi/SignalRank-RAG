from signalrank.components.chunking.chunking import DocumentChunker
from signalrank.components.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from signalrank.components.retrieval.base import Retriever
from signalrank.components.retrieval.bm25 import BM25Retriever
from signalrank.components.retrieval.dense import DenseRetriever
from signalrank.components.vector_store.qdrant import QdrantVectorStore
from signalrank.config.configuration import ConfigurationManager
from signalrank.pipelines.data_ingestion_pipeline import DataIngestionPipeline


class RetrievalPipeline:
    """
    Build the configured retrieval stack.
    """

    def build(self) -> tuple[Retriever, Retriever]:
        config_manager = ConfigurationManager()

        # 1. Ingest
        documents = DataIngestionPipeline().run()

        # 2. Chunk
        chunking_config = (
            config_manager.get_chunking_config()
        )

        chunker = DocumentChunker(
            config=chunking_config,
        )

        chunks = chunker.chunk_documents(documents)

        # 3. BM25

        bm25 = BM25Retriever(chunks)

        # 4. Embeddings

        embedding_provider = SentenceTransformerEmbedding()

        vectors = embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )

        # 5. Vector store

        vector_store = QdrantVectorStore(
            dimension=embedding_provider.dimension,
        )

        vector_store.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            vectors=vectors,
            payloads=[
                {
                    "doc_id": chunk.doc_id,
                    "source_path": chunk.source_path,
                }
                for chunk in chunks
            ],
        )

        # 6. Dense retriever

        chunk_map = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

        dense = DenseRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            chunks=chunk_map,
        )

        return bm25, dense