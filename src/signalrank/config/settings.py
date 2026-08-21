from dataclasses import dataclass
from pathlib import Path

from typing import Literal

from signalrank.config.settings_utils import normalise_supported_extensions
from signalrank.components.embeddings.base import EmbeddingProviderType

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
            normalise_supported_extensions(
                self.supported_extensions
                )
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
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding provider."""

    provider: EmbeddingProviderType = "sentence_transformer"
    model_name: str | None = None
    dimension: int | None = None

    def __post_init__(self) -> None:
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty"
            )

        if self.dimension is not None and self.dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero"
            )


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval."""

    top_k: int = 5

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )


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
    path: Path | None = Path("data/qdrant")

    def __post_init__(self) -> None:
        if self.path is not None:
            object.__setattr__(
                self,
                "path",
                Path(self.path),
            )

        if not self.collection_name.strip():
            raise ValueError(
                "collection_name cannot be empty"
            )

        if self.mode == "local" and self.path is None:
            raise ValueError(
                "path is required when Qdrant mode is 'local'"
            )


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
            raise ValueError(
                "language cannot be empty"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

@dataclass(frozen=True)
class AppConfig:
    data_ingestion: DataIngestionConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    qdrant: QdrantConfig
    logfire: LogfireConfig
    finewiki: FineWikiConfig