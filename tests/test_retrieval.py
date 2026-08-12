from signalrank.components.chunking.chunk import DocumentChunk
from signalrank.components.retrieval.bm25 import BM25Retriever

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

