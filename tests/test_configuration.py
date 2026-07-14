from signalrank.config.configuration import ConfigurationManager


def test_default_configuration_loading():
    config = ConfigurationManager().get_config()

    ingestion = config["data_ingestion"]
    embedding = config["embedding"]
    chunking = config["chunking"]
    retrieval = config["retrieval"]

    assert ingestion["source_path"] == "data/"
    assert ingestion["recursive"] is True
    assert ingestion["encoding"] == "utf-8"
    assert ingestion["supported_extensions"] == [
        ".txt",
        ".md",
        ".markdown",
    ]

    assert embedding["model_name"] == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    assert chunking["chunk_size"] == 1000
    assert chunking["chunk_overlap"] == 200
    assert retrieval["top_k"] == 5