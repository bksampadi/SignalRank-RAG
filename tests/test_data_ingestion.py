import pytest

from signalrank.components.data_ingestion import DataIngestion
from signalrank.entity.config_entity import DataIngestionConfig
from signalrank.exception.exception import SignalRankException

def test_ingests_single_text_file(tmp_path):
    source_file = tmp_path / "document.txt"
    source_file.write_text(
        "SignalRank evaluates retrieval signals.",
        encoding="utf-8",
    )

    config = DataIngestionConfig(source_path=source_file)
    artifact = DataIngestion(config).initiate_data_ingestion()

    assert artifact.total_documents == 1
    assert len(artifact.documents) == 1

    document = artifact.documents[0]

    assert document.source_path == "document.txt"
    assert document.text == "SignalRank evaluates retrieval signals."
    assert document.doc_id.startswith("doc_")

    assert document.metadata["source"] == "document.txt"
    assert document.metadata["file_name"] == "document.txt"
    assert document.metadata["file_extension"] == ".txt"
    assert document.metadata["char_count"] == len(document.text)
    assert document.metadata["word_count"] == 4


def test_ingests_supported_files_recursively(tmp_path):
    corpus = tmp_path / "corpus"
    nested = corpus / "nested"
    nested.mkdir(parents=True)

    (corpus / "first.md").write_text(
        "First document",
        encoding="utf-8",
    )
    (nested / "second.txt").write_text(
        "Second document",
        encoding="utf-8",
    )
    (nested / "ignored.pdf").write_bytes(b"Not a supported document")

    config = DataIngestionConfig(
        source_path=corpus,
        recursive=True,
    )

    artifact = DataIngestion(config).initiate_data_ingestion()

    assert artifact.total_documents == 2

    source_paths = {
        document.source_path
        for document in artifact.documents
    }

    assert source_paths == {
        "first.md",
        "nested/second.txt",
    }

def test_non_recursive_ingestion_ignores_nested_files(tmp_path):
    corpus = tmp_path / "corpus"
    nested = corpus / "nested"
    nested.mkdir(parents=True)

    (corpus / "root.txt").write_text(
        "Root document",
        encoding="utf-8",
    )
    (nested / "nested.txt").write_text(
        "Nested document",
        encoding="utf-8",
    )

    config = DataIngestionConfig(
        source_path= corpus,
        recursive=False,
    )

    artifact = DataIngestion(config).initiate_data_ingestion()

    assert artifact.total_documents == 1
    assert artifact.documents[0].source_path == "root.txt"


def test_skips_empty_documents(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "content.txt").write_text(
        "Document with content",
        encoding="utf-8",
    )
    (corpus / "empty.txt").write_text(
        "   \n",
        encoding="utf-8",
    )

    config = DataIngestionConfig(source_path=corpus)
    artifact = DataIngestion(config).initiate_data_ingestion()

    assert artifact.total_documents == 1
    assert artifact.documents[0].source_path == "content.txt"


def test_document_id_is_portable_and_content_sensitive(tmp_path):
    first_corpus = tmp_path / "first"
    second_corpus = tmp_path / "second"

    first_corpus.mkdir()
    second_corpus.mkdir()

    first_file = first_corpus / "article.txt"
    second_file = second_corpus / "article.txt"

    first_file.write_text("Identical content", encoding="utf-8")
    second_file.write_text("Identical content", encoding="utf-8")

    first_artifact = DataIngestion(
        DataIngestionConfig(source_path=first_corpus)
    ).initiate_data_ingestion()

    second_artifact = DataIngestion(
        DataIngestionConfig(source_path=second_corpus)
    ).initiate_data_ingestion()

    first_id = first_artifact.documents[0].doc_id
    second_id = second_artifact.documents[0].doc_id

    # Absolute root directories differ, but the portable source
    # reference and document content are identical.

    assert first_id == second_id

    second_file.write_text("Changed content", encoding="utf-8")

    changed_artifact = DataIngestion(
        DataIngestionConfig(source_path=second_corpus)
    ).initiate_data_ingestion()

    assert changed_artifact.documents[0].doc_id != first_id

def test_normalises_configured_extensions(tmp_path):
    source_file = tmp_path / "document.TXT"
    source_file.write_text("Supported content", encoding="utf-8")

    config = DataIngestionConfig(
        source_path=source_file,
        supported_extensions=("TXT",),
    )

    artifact = DataIngestion(config).initiate_data_ingestion()

    assert artifact.total_documents == 1


def test_missing_source_raises_signalrank_exception(tmp_path):
    missing_source = tmp_path / "missing"

    config = DataIngestionConfig(source_path=missing_source)

    with pytest.raises(SignalRankException):
        DataIngestion(config).initiate_data_ingestion()


def test_all_empty_documents_raise_signalrank_exception(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "empty.txt").write_text("", encoding="utf-8")
    (corpus / "whitespace.md").write_text(" \n\t", encoding="utf-8")

    config = DataIngestionConfig(source_path=corpus)

    with pytest.raises(SignalRankException):
        DataIngestion(config).initiate_data_ingestion()
