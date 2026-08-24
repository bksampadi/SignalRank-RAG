from pathlib import Path

import logfire
from qdrant_client import QdrantClient

from signalrank.components.chunking.chunking import DocumentChunker
from signalrank.components.embeddings.factory import create_embedding_provider
from signalrank.components.retrieval.base import Retriever
from signalrank.components.retrieval.bm25 import BM25Retriever
from signalrank.components.retrieval.dense import DenseRetriever
from signalrank.components.retrieval.hybrid import HybridRetriever
from signalrank.components.vector_store.qdrant import QdrantVectorStore
from signalrank.config.configuration import ConfigurationManager
from signalrank.constants import CONFIG_FILE_PATH
from signalrank.pipelines.data_ingestion_pipeline import (
    DataIngestionPipeline,
)


class RetrievalPipeline:
    """
    Build the configured retrieval stack.

    The dense vector index must already exist.
    """

    def __init__(
        self,
        config_filepath: str | Path = CONFIG_FILE_PATH,
    ):
        self._config_filepath = config_filepath

    def build(
        self,
    ) -> tuple[Retriever, Retriever, Retriever]:
        config = ConfigurationManager(self._config_filepath).load()

        with logfire.span(
            "Build retrieval pipeline",
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model_name,
            collection_name=config.qdrant.collection_name,
        ):
            # 1. Load documents

            documents = DataIngestionPipeline(
                config_filepath=self._config_filepath,
            ).run()

            # 2. Reconstruct deterministic chunks

            chunker = DocumentChunker(
                config=config.chunking,
            )

            chunks = chunker.chunk_documents(documents)

            # 3. Build lexical retriever

            bm25 = BM25Retriever(chunks)

            # 4. Create configured embedding provider

            embedding_provider = create_embedding_provider(
                provider=config.embedding.provider,
                model_name=config.embedding.model_name,
                dimension=config.embedding.dimension,
            )

            # 5. Connect to existing Qdrant index

            qdrant_client = QdrantClient(
                path=str(config.qdrant.path),
            )

            if not qdrant_client.collection_exists(config.qdrant.collection_name):
                qdrant_client.close()

                raise RuntimeError(
                    "Qdrant collection "
                    f"'{config.qdrant.collection_name}' "
                    "does not exist. Run the indexing pipeline "
                    "before starting retrieval."
                )

            vector_store = QdrantVectorStore(
                dimension=embedding_provider.dimension,
                collection_name=config.qdrant.collection_name,
                client=qdrant_client,
            )

            # 6. Build deterministic chunk lookup

            chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

            # 7. Build dense retriever

            dense = DenseRetriever(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                chunks=chunk_map,
            )

            # 8. Build hybrid retriever

            hybrid = HybridRetriever(
                bm25_retriever=bm25,
                dense_retriever=dense,
                rrf_k=config.retrieval.rrf_k,
                candidate_multiplier=config.retrieval.candidate_multiplier,
            )

            return bm25, dense, hybrid
