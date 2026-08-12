from signalrank.components.chunking.chunk import DocumentChunk
from signalrank.components.retrieval.bm25 import BM25Retriever
from signalrank.components.retrieval.dense import DenseRetriever

def make_chunk(
        text: str,
        index: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk_{index}",
        doc_id="doc_test",
        source_path="text.txt",
        file_type=".txt",
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len(text),
        element_indices=(index,),
        element_types=("text",),
    )

def test_bm25_returns_most_relevant_chunk():

    chunks = [
        make_chunk(
            "The Mars rover collected rock samples from the crater.",
            0,
        ),
        make_chunk(
            "The Atlantic Ocean contains several major currents.",
            1,
        ),
        make_chunk(
            "A rover mission to Mars requires autonomous navigation.",
            2,
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.retrieve(
        "Mars rover geological samples"
    )

    assert results[0].chunk_id == chunks[0].chunk_id


class FakeEmbeddingProvider:
    @property
    def dimension(self)-> int:
        return 3

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_documents(
            self,
            texts: list[str],
    ) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def search(
            self,
            query_vector: list[float],
            top_k: int = 10,
    ) -> list[tuple[str, float]]:
        return[
            ("chunk_1", 0.92),
            ("chunk_2", 0.81),
        ]


def make_dense_chunk(
        chunk_id: str,
        text: str,
        index: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id="doc_test",
        source_path="text.txt",
        file_type=".txt",
        chunk_index=index,
        text=text,
        char_start=0,
        char_end=len(text),
        element_indices=(index,),
        element_types=("text",),
    )


def test_dense_retriever_returns_ranked_results():
    chunks = {
        "chunk_1": make_dense_chunk(
            "chunk_1",
            "The Mars rover collected geological samples.",
            0,
        ),
        "chunk_2": make_dense_chunk(
            "chunk_2",
            "Robotic spacecraft explore distant planets.",
            1,
        ),
    }

    retriever = DenseRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=FakeVectorStore(),
        chunks=chunks,
    )

    results = retriever.retrieve(
        "vehicle exploring the red planet"
    )

    assert len(results) == 2

    assert results[0].chunk_id == "chunk_1"
    assert results[0].score == 0.92
    assert results[0].rank == 1

    assert results[1].chunk_id == "chunk_2"
    assert results[1].score == 0.81
    assert results[1].rank == 2