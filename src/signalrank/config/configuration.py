from signalrank.constants import *
from signalrank.config.settings import DataIngestionConfig
from signalrank.utils.common import read_yaml

from box import ConfigBox


class ConfigurationManager:
    """Load SignalRank configuration from a file or packaged defaults."""

    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)

    def get_config(self):
        return self.config
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        data_ingestion_config = DataIngestionConfig(
            source_path=config.source_path,
            recursive=config.recursive,
            encoding=config.encoding,
            supported_extensions=config.supported_extensions,
        )

        return data_ingestion_config
