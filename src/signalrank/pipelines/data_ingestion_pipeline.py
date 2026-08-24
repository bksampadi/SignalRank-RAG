from pathlib import Path

from signalrank.components.data_ingestion.data_ingestion import DataIngestion
from signalrank.components.data_ingestion.document import ParsedDocument
from signalrank.config.configuration import ConfigurationManager
from signalrank.constants import CONFIG_FILE_PATH


class DataIngestionPipeline:
    """Orchestrate configured document ingestion."""

    def __init__(
        self,
        config_filepath: str | Path = CONFIG_FILE_PATH,
    ):
        self._config_filepath = Path(config_filepath)

    def run(self) -> list[ParsedDocument]:
        config = ConfigurationManager(self._config_filepath).load()

        ingestion = DataIngestion(
            config=config.data_ingestion,
        )

        return ingestion.initiate_data_ingestion()
