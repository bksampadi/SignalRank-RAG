from __future__ import annotations

from signalrank.components.data_ingestion.data_ingestion import DataIngestion
from signalrank.components.data_ingestion.document import ParsedDocument
from signalrank.config.configuration import ConfigurationManager


class DataIngestionPipeline:
    """Orchestrate configured document ingestion."""

    def run(self) -> list[ParsedDocument]:
        config_manager = ConfigurationManager()

        ingestion_config = (
            config_manager.get_data_ingestion_config()
        )

        ingestion = DataIngestion(
            config=ingestion_config,
        )

        return ingestion.initiate_data_ingestion()