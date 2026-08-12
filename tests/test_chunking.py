from signalrank.components.chunking.chunking import DocumentChunker
from signalrank.components.data_ingestion.document import (
    DocumentElement,
    ParsedDocument,
)
from signalrank.config.settings import ChunkingConfig
from signalrank.config.configuration import ConfigurationManager


def make_document(text: str) -> ParsedDocument:
        return ParsedDocument(
            doc_id="doc_test",
            source_path="test.txt",
            file_type=".txt",
            elements=(
                DocumentElement(
                    text=text,
                    element_type="text",
                    element_index= 0,
                ),
            ),
        )
    


def test_short_document_produces_single_chunk():
    chunker = DocumentChunker(
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
        )
    )

    chunks = chunker.chunk_document(
        make_document("hello world")
    )

    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].chunk_index == 0


def test_chunk_overlap_is_preserved():
    chunker = DocumentChunker(
        ChunkingConfig(
            chunk_size=6,
            chunk_overlap=2,
        )
    )

    chunks = chunker.chunk_document(
        make_document("abcdefghij")
    )

    assert len(chunks) == 2
    assert chunks[0].text == "abcdef"
    assert chunks[1].text == "efghij"


def test_chunk_ids_are_deterministic():
    chunker = DocumentChunker(
        ChunkingConfig(
            chunk_size=6,
            chunk_overlap=2,
        )
    )

    document = make_document("abcdefghij")

    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_chunk_preserves_document_provenance():
    document = ParsedDocument(
        doc_id="doc_test",
        source_path="paper.txt",
        file_type=".txt",
        elements=(
            DocumentElement(
                text="first paragraph",
                element_type="paragraph",
                element_index=0,
            ),
            DocumentElement(
                text="second paragraph",
                element_type="paragraph",
                element_index=1,
            ),
        ),
    )

    chunker = DocumentChunker(
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
        )
    )

    chunks = chunker.chunk_document(document)

    assert chunks[0].doc_id == "doc_test"
    assert chunks[0].source_path == "paper.txt"
    assert chunks[0].element_indices == (0, 1)
    assert chunks[0].element_types == (
        "paragraph",
        "paragraph",
    )


def test_empty_elements_produce_no_chunks():
    document = ParsedDocument(
        doc_id="doc_test",
        source_path="empty.txt",
        file_type=".txt",
        elements=(
            DocumentElement(
                text="   ",
                element_type="text",
                element_index=0,
            ),
        ),
    )

    chunker = DocumentChunker(
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
        )
    )

    assert chunker.chunk_document(document) == []



def test_chunking_configuration_loading():
    config = ConfigurationManager().load()

    assert config.chunking.chunk_size == 1000
    assert config.chunking.chunk_overlap == 200