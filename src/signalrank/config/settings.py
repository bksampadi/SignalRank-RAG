from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from signalrank.components.embeddings.base import EmbeddingProviderType
from signalrank.config.settings_utils import normalise_supported_extensions


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for the data ingestion component."""

    source_path: Path
    recursive: bool = True
    encoding: str = "utf-8"
    supported_extensions: tuple[str, ...] = (
        ".pdf",
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".docx",
        ".pptx",
        ".xlsx",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "supported_extensions",
            normalise_supported_extensions(self.supported_extensions),
        )


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for deterministic text chunking."""

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding provider."""

    provider: EmbeddingProviderType = "sentence_transformer"
    model_name: str | None = None
    dimension: int | None = None

    def __post_init__(self) -> None:
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("dimension must be greater than zero")


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval."""

    top_k: int = 5
    rrf_k: int = 60
    candidate_multiplier: int = 4

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")


RankingProvider = Literal["flashrank",]


@dataclass(frozen=True)
class RankingConfig:
    """Configuration for second-stage ranking."""

    enabled: bool = True
    provider: RankingProvider = "flashrank"
    model_name: str = "ms-marco-MiniLM-L-12-v2"
    max_length: int = 512
    candidate_multiplier: int = 4

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider cannot be empty")

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

        if self.max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")


QdrantMode = Literal[
    "memory",
    "local",
    "remote",
]


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for the Qdrant vector store."""

    mode: QdrantMode = "memory"
    collection_name: str = "signalrank_finewiki"
    recreate_collection: bool = False
    path: Path | None = Path("data/qdrant")

    def __post_init__(self) -> None:
        if self.path is not None:
            object.__setattr__(
                self,
                "path",
                Path(self.path),
            )

        if not self.collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        if self.mode == "local" and self.path is None:
            raise ValueError("path is required when Qdrant mode is 'local'")


LLMProviderType = Literal[
    "openrouter",
    "groq",
]


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for the LLM layer."""

    provider: LLMProviderType = "openrouter"
    model_name: str = "openrouter/free"
    fallback_models: tuple[str, ...] = ()
    max_retries: int = 1
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fallback_models",
            tuple(self.fallback_models),
        )

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")


@dataclass(frozen=True)
class LogfireConfig:
    """Configuration for Logfire observability."""

    service_name: str = "signalrank-rag"
    environment: str = "local"
    system_metrics: bool = False


@dataclass(frozen=True)
class FineWikiConfig:
    """Configuration for FineWiki ingestion."""

    language: str = "enwiki"
    batch_size: int = 256

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("language cannot be empty")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")


@dataclass(frozen=True)
class AppConfig:
    data_ingestion: DataIngestionConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    ranking: RankingConfig
    qdrant: QdrantConfig
    llm: LLMConfig | None
    logfire: LogfireConfig
    finewiki: FineWikiConfig
