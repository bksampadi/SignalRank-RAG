import logfire
from qdrant_client import QdrantClient

from signalrank.components.chunking.chunk import DocumentChunk
from signalrank.components.chunking.chunking import DocumentChunker
from signalrank.components.embeddings.factory import create_embedding_provider
from signalrank.components.vector_store.qdrant import QdrantVectorStore
from signalrank.config.configuration import ConfigurationManager
from signalrank.pipelines.data_ingestion_pipeline import (
    DataIngestionPipeline,
)


class IndexingPipeline:
    """
    Build the configured document index.

    Ingest -> chunk -> embed -> store in Qdrant.
    """

    def run(self) -> list[DocumentChunk]:
        config = ConfigurationManager().load()

        with logfire.span(
            "Indexing pipeline",
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model_name,
            collection_name=config.qdrant.collection_name,
        ) as span:

            # 1. Ingest documents

            documents = DataIngestionPipeline().run()

            span.set_attribute(
                "document_count",
                len(documents),
            )

            # 2. Chunk documents

            chunker = DocumentChunker(
                config=config.chunking,
            )

            chunks = chunker.chunk_documents(
                documents
            )

            span.set_attribute(
                "chunk_count",
                len(chunks),
            )

            if not chunks:
                return []

            # 3. Create configured embedding provider

            embedding_provider = create_embedding_provider(
                provider=config.embedding.provider,
                model_name=config.embedding.model_name,
                dimension=config.embedding.dimension,
            )

            # 4. Embed document chunks

            vectors = embedding_provider.embed_documents(
                [
                    chunk.text
                    for chunk in chunks
                ]
            )

            # 5. Open vector store

            qdrant_client = QdrantClient(
                path=str(config.qdrant.path),
            )

            try:
                vector_store = QdrantVectorStore(
                    dimension=embedding_provider.dimension,
                    collection_name=config.qdrant.collection_name,
                    client=qdrant_client,
                )

                # 6. Index vectors

                vector_store.upsert(
                    ids=[
                        chunk.chunk_id
                        for chunk in chunks
                    ],
                    vectors=vectors,
                    payloads=[
                        {
                            "doc_id": chunk.doc_id,
                            "source_path": chunk.source_path,
                            "file_type": chunk.file_type,
                            "chunk_index": chunk.chunk_index,
                            "text": chunk.text,
                        }
                        for chunk in chunks
                    ],
                )

                span.set_attribute(
                    "indexed_vector_count",
                    len(vectors),
                )

            finally:
                qdrant_client.close()

            return chunks