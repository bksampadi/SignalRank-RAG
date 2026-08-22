import pytest

from signalrank.components.data_ingestion.data_ingestion import DataIngestion
from signalrank.config.settings import DataIngestionConfig


def test_ingests_single_text_file(tmp_path):
    source_file = tmp_path / "document.txt"
    source_file.write_text(
        "SignalRank evaluates retrieval signals.",
        encoding="utf-8",
    )

    config = DataIngestionConfig(
        source_path=source_file,
        recursive=True,
        encoding="utf-8",
    )

    documents = DataIngestion(config).initiate_data_ingestion()

    assert len(documents) == 1

    document = documents[0]

    assert document.source_path == "document.txt"
    assert document.file_type == ".txt"
    assert document.doc_id.startswith("doc_")
    assert document.metadata["element_count"] == 1

    element = document.elements[0]

    assert element.text == "SignalRank evaluates retrieval signals."
    assert element.element_type == "text"
    assert element.element_index == 0
    assert element.metadata["extension"] == ".txt"


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

    # Deliberately unsupported.
    (nested / "ignored.xyz").write_text(
        "Ignore me",
        encoding="utf-8",
    )

    config = DataIngestionConfig(
        source_path=corpus,
        recursive=True,
    )

    documents = DataIngestion(config).initiate_data_ingestion()

    assert len(documents) == 2

    source_paths = {document.source_path for document in documents}

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
        source_path=corpus,
        recursive=False,
    )

    documents = DataIngestion(config).initiate_data_ingestion()

    assert len(documents) == 1
    assert documents[0].source_path == "root.txt"


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

    config = DataIngestionConfig(
        source_path=corpus,
    )

    documents = DataIngestion(config).initiate_data_ingestion()

    assert len(documents) == 1
    assert documents[0].source_path == "content.txt"


def test_document_id_is_portable_and_content_sensitive(tmp_path):
    first_corpus = tmp_path / "first"
    second_corpus = tmp_path / "second"

    first_corpus.mkdir()
    second_corpus.mkdir()

    first_file = first_corpus / "article.txt"
    second_file = second_corpus / "article.txt"

    first_file.write_text(
        "Identical content",
        encoding="utf-8",
    )

    second_file.write_text(
        "Identical content",
        encoding="utf-8",
    )

    first_documents = DataIngestion(
        DataIngestionConfig(
            source_path=first_corpus,
        )
    ).initiate_data_ingestion()

    second_documents = DataIngestion(
        DataIngestionConfig(
            source_path=second_corpus,
        )
    ).initiate_data_ingestion()

    first_id = first_documents[0].doc_id
    second_id = second_documents[0].doc_id

    # Absolute root directories differ, but the portable source
    # reference and document content are identical.
    assert first_id == second_id

    second_file.write_text(
        "Changed content",
        encoding="utf-8",
    )

    changed_documents = DataIngestion(
        DataIngestionConfig(
            source_path=second_corpus,
        )
    ).initiate_data_ingestion()

    assert changed_documents[0].doc_id != first_id


def test_normalises_configured_extensions(tmp_path):
    source_file = tmp_path / "document.TXT"
    source_file.write_text(
        "Supported content",
        encoding="utf-8",
    )

    config = DataIngestionConfig(
        source_path=source_file,
        supported_extensions=("TXT",),
    )

    documents = DataIngestion(config).initiate_data_ingestion()

    assert len(documents) == 1
    assert documents[0].file_type == ".txt"


def test_missing_source_raises_file_not_found_error(tmp_path):
    missing_source = tmp_path / "missing"

    config = DataIngestionConfig(
        source_path=missing_source,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Source path does not exist",
    ):
        DataIngestion(config).initiate_data_ingestion()


def test_all_empty_documents_raise_value_error(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "empty.txt").write_text(
        "",
        encoding="utf-8",
    )

    (corpus / "whitespace.md").write_text(
        " \n\t",
        encoding="utf-8",
    )

    config = DataIngestionConfig(
        source_path=corpus,
    )

    with pytest.raises(
        ValueError,
        match="Data ingestion produced no documents",
    ):
        DataIngestion(config).initiate_data_ingestion()


def test_unsupported_single_file_raises_value_error(tmp_path):
    source_file = tmp_path / "document.xyz"
    source_file.write_text(
        "Unsupported content",
        encoding="utf-8",
    )

    config = DataIngestionConfig(
        source_path=source_file,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported file type",
    ):
        DataIngestion(config).initiate_data_ingestion()
