from dataclasses import dataclass
from pathlib import Path
from ensure import ensure_annotations
from signalrank.config.settings_utils import normalise_supported_extensions


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for the data ingestion component."""

    source_path: Path
    recursive: bool = True
    encoding: str = "utf-8"
    supported_extensions: tuple[str, ...] = (
        ".txt",
        ".md",
        ".markdown",
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