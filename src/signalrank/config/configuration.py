from __future__ import annotations

from box import ConfigBox

from signalrank.constants import CONFIG_FILE_PATH
from signalrank.config.settings import DataIngestionConfig
from signalrank.utils.common import read_yaml


class ConfigurationManager:
    """Load SignalRank configuration from a file or packaged defaults."""

    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)

    def get_config(self) -> ConfigBox:
        """Return the complete raw configuration."""

        return self.config
    
    def get_data_ingestion_config(
            self,
    ) -> DataIngestionConfig:
        """Return validated data-ingestion configuration."""
        
        config = self.config.data_ingestion
        
        return DataIngestionConfig(
            source_path=config.source_path,
            recursive=config.recursive,
            encoding=config.encoding,
            supported_extensions=tuple(
                config.supported_extensions
            ),
        )
