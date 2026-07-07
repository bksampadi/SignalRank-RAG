from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for the data ingestion component."""

    source_path: Path
    recursive: bool = True
    encoding: str = "utf-8"
    supported_extensions: tuple[str, ...] = (".txt", ".md", ".markdown")
    