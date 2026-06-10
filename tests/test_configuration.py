from src.signalrank.config.configuration import ConfigurationManager

def test_configuration_loading():

    config = ConfigurationManager().get_config()

    assert config["data"]["docs_path"] == "data/docs.txt"
    assert config["retrieval"]["top_k"] == 5
    assert (
        config["embedding"]["model_name"]
        == "sentence-transformers/all-MiniLM-L6-v2"
    )