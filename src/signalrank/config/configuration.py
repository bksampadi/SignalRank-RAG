from signalrank.config.settings import (
    AppConfig,
    ChunkingConfig,
    DataIngestionConfig,
    EmbeddingConfig,
    FineWikiConfig,
    LogfireConfig,
    QdrantConfig,
    RetrievalConfig,
)
from signalrank.constants import CONFIG_FILE_PATH
from signalrank.utils.common import read_yaml


class ConfigurationManager:
    """Load SignalRank configuration from a file or packaged defaults."""

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)

    def load(self) -> AppConfig:
        return AppConfig(
            data_ingestion=DataIngestionConfig(**self.config.data_ingestion),
            chunking=ChunkingConfig(**self.config.chunking),
            embedding=EmbeddingConfig(**self.config.embedding),
            retrieval=RetrievalConfig(**self.config.retrieval),
            qdrant=QdrantConfig(**self.config.qdrant),
            logfire=LogfireConfig(**self.config.logfire),
            finewiki=FineWikiConfig(**self.config.finewiki),
        )
