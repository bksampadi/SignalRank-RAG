from collections.abc import Iterator
from pathlib import Path

import logfire

from signalrank.components.data_ingestion.document import ParsedDocument
from signalrank.components.data_ingestion.registry import get_loader
from signalrank.config.settings import DataIngestionConfig
from signalrank.utils.common import create_document_id


class DataIngestion:
    """Discover supported files and load them into normalized documents."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.source_path = config.source_path
        self.supported_extensions = config.supported_extensions

    def initiate_data_ingestion(self) -> list[ParsedDocument]:
        """
        Load supported files into normalized parsed documents.
        """
        with logfire.span(
            "Data ingestion",
            source_path=str(self.source_path),
            recursive=self.config.recursive,
            supported_extensions=self.supported_extensions,
        ) as span:
            try:
                files = list(self._collect_files())

                span.set_attribute(
                    "files_discovered",
                    len(files),
                )

                if not files:
                    raise FileNotFoundError(
                        f"No supported files found at {self.source_path}. "
                        f"Supported extensions: {self.supported_extensions}"
                    )

                documents: list[ParsedDocument] = []

                for file_path in files:
                    source_reference = self._get_source_reference(file_path)

                    loader = get_loader(
                        file_path.suffix.lower(), encoding=self.config.encoding
                    )

                    elements = loader(file_path)

                    if not elements:
                        logfire.warning(
                            "No extractable content found",
                            file_path=str(file_path),
                        )
                        continue

                    combined_text = "\n".join(element.text for element in elements)

                    documents.append(
                        ParsedDocument(
                            doc_id=create_document_id(
                                source_reference,
                                combined_text,
                            ),
                            source_path=source_reference,
                            file_type=file_path.suffix.lower(),
                            elements=tuple(elements),
                            metadata={
                                "element_count": len(elements),
                            },
                        )
                    )

                if not documents:
                    raise ValueError("Data ingestion produced no documents")

                span.set_attribute(
                    "documents_loaded",
                    len(documents),
                )

                return documents

            except Exception:
                logfire.exception(
                    "Data ingestion failed",
                    source_path=str(self.source_path),
                )
                raise

    def _get_source_reference(self, file_path: Path) -> str:
        """
        Return a portable path relative to the configured source.
        """

        if self.source_path.is_dir():
            return file_path.relative_to(self.source_path).as_posix()

        return file_path.name

    def _collect_files(self) -> Iterator[Path]:
        """Collect supported files from the configured source."""

        if not self.source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.source_path}")

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
