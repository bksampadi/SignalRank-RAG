from signalrank.components.chunking.chunk import DocumentChunk
from signalrank.components.embeddings.sentence_transformer import (
    SentenceTransformerEmbedding,
)
from signalrank.components.retrieval.dense import DenseRetriever
from signalrank.components.vector_store.qdrant import QdrantVectorStore


def make_chunk(
        chunk_id: str,
        text: str,
        index: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc_test",
        source_path="test.txt",
        file_type=".txt",
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len("text"),
        element_indices=(index,),
        element_types=("text",),
    )

def test_real_dense_retrieval_returns_semantic_match():
    chunks = [
        make_chunk(
            "chunk_mars",
            "The Mars rover collected geological samples from the planet surface.",
            0,
        ),
        make_chunk(
            "chunk_ocean",
            "Ocean currents transport heat across the Atlantic",
            1,
        ),
        make_chunk(
            "chunk_weather",
            "Weather stations measure rainfall and atmospheric temperature.",
            2,
        ),
    ]

    embedding_provider = SentenceTransformerEmbedding()

    vectors = embedding_provider.embed_documents(
        [chunk.text for chunk in chunks]
    )

    vector_store = QdrantVectorStore(
        dimension=embedding_provider.dimension,
        collection_name="dense_integration_test",
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

    chunk_map = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    retriever = DenseRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunks=chunk_map,
    )

    results = retriever.retrieve(
        "robotic vehicle exploring the red planet",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].chunk_id == "chunk_mars"
    assert results[0].rank == 1
    assert isinstance(results[0].score, float)