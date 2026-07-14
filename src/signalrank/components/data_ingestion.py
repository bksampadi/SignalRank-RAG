from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

from src.signalrank.entity.config_entity import DataIngestionConfig
from src.signalrank.entity.artifact_entity import (
    DataIngestionArtifact,
    IngestedDocument,
    )

from src.signalrank.exception.exception import SignalRankException
from src.signalrank.logging.logger import logging
from src.signalrank.utils.common import (
    create_document_id,
    get_file_metadata,
    read_text_file,
)

class DataIngestion:
    """
    Component responsible for loading raw documents from disk.

    This component does not process the raw documents further.
    """

    def __init__(self, config:DataIngestionConfig):
        self.config = config
        self.source_path = Path(config.source_path)
        self.supported_extensions = tuple(
            ext.lower() for ext in config.supported_extensions
        )

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Read supported files and return them as IngestedDocument objects"""
        try:
            logging.info("Starting data ingestion from: %s", self.source_path)

            files= list(self._collect_files())

            if not files:
                raise FileNotFoundError(
                    f"No supported files found at {self.source_path}. "
                    f"Supported extensions: {self.supported_extensions}"
                )
            
            documents: list[IngestedDocument] = []

            for file_path in files:
                text = read_text_file(
                    file_path=file_path,
                    encoding=self.config.encoding,
                    )

                if not text.strip():
                    logging.warning("Skipping empty document: %s", file_path)
                    continue

                document = IngestedDocument(
                    doc_id=create_document_id(file_path),
                    source_path=str(file_path),
                    text=text,
                    metadata=get_file_metadata(file_path, text),
                )

                documents.append(document)

            if not documents:
                raise ValueError(
                    "Data Ingestion found files, but all documents were empty"
                )
            
            artifact = DataIngestionArtifact(
                documents=documents,
                total_documents=len(documents),
            )
            
            logging.info(
                "Data ingestion completed. Loaded %d documents.",
                artifact.total_documents,
            )

            return artifact

        except Exception as e:
            raise SignalRankException(e, sys)
        
    def _collect_files(self) -> Iterator[Path]:
        """Collect supported files from source path."""
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Source path does not exist: {self.source_path}"
            )
        
        if self.source_path.is_file():
            if self._is_supported_file(self.source_path):
                yield self.source_path
                return
            
            raise ValueError(
                f"Unsupported file type: {self.source_path.suffix}. "
                f"Supported extensions: {self.supported_extensions}"
            )
        
        if self.source_path.is_dir():
            pattern = "**/*" if self.config.recursive else "*"

            for file_path in sorted(self.source_path.glob(pattern)):
                if file_path.is_file() and self._is_supported_file(file_path):
                    yield file_path

            return
        
        raise ValueError(
            f"Source path is neither a file nor a directory: {self.source_path}"
        )
    
    def _is_supported_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions
    