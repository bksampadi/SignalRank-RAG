from signalrank.config.configuration import ConfigurationManager
from signalrank.components.data_ingestion import DataIngestion
from signalrank.config.settings import DataIngestionConfig
from signalrank.logging.logger import logging
from signalrank.exception.exception import SignalRankException


class DataIngestionTrainingPipeline:
    def __init__(
            self
    ):
        pass
    def initiate_data_ingestion(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_config
        data_ingestion = DataIngestion(config=DataIngestionConfig)

        data_ingestion.initiate_data_ingestion()
