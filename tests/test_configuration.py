import pytest

from signalrank.config.configuration import ConfigurationManager, DataIngestionConfig


def test_default_configuration_loading():
    config = ConfigurationManager().load()

    assert config.data_ingestion.source_path.name == "data"
    assert config.chunking.chunk_size == 1000
    assert config.chunking.chunk_overlap == 200
    assert config.embedding.provider == "sentence_transformer"
    assert config.embedding.model_name == "sentence-transformers/all-mpnet-base-v2"
    assert config.retrieval.top_k == 5
    assert config.qdrant.mode == "local"
    assert config.qdrant.collection_name == "signalrank_finewiki"
    assert config.qdrant.path is not None
    assert config.qdrant.path.as_posix() == "data/qdrant"
    assert config.logfire.service_name == "signalrank-rag"
    assert config.logfire.environment == "local"
    assert config.finewiki.language == "enwiki"
    assert config.finewiki.batch_size == 256


def test_invalid_extension_configuration_raises_value_error(
    tmp_path,
):
    with pytest.raises(ValueError):
        DataIngestionConfig(
            source_path=tmp_path,
            supported_extensions=("",),
        )


def test_data_ingestion_config_normalises_extensions(
    tmp_path,
):
    config = DataIngestionConfig(
        source_path=tmp_path,
        recursive=True,
        encoding="utf-8",
    )

    assert config.source_path == tmp_path
