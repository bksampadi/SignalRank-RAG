from signalrank.components.data_ingestion.data_ingestion import DataIngestion
from signalrank.components.data_ingestion.document import ParsedDocument
from signalrank.config.configuration import ConfigurationManager


class DataIngestionPipeline:
    """Orchestrate configured document ingestion."""

    def run(self) -> list[ParsedDocument]:
        config = ConfigurationManager().load()

        ingestion = DataIngestion(
            config=config.data_ingestion,
        )

        return ingestion.initiate_data_ingestion()