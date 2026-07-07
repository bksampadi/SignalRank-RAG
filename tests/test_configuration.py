from src.signalrank.config.configuration import ConfigurationManager


def test_configuration_loading():
    config = ConfigurationManager().get_config()

    assert config["data_ingestion"]["source_path"] == "data/"
    assert config["data_ingestion"]["recursive"] is True
    assert config["data_ingestion"]["encoding"] == "utf-8"
    assert config["data_ingestion"]["supported_extensions"] == [
        ".txt",
        ".md",
        ".markdown",
    ]

    assert config["embedding"]["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert config["chunking"]["chunk_size"] == 1000
    assert config["chunking"]["chunk_overlap"] == 200
    assert config["retrieval"]["top_k"] == 5