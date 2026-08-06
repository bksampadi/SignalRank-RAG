from dataclasses import dataclass

@dataclass(frozen=True)
class IngestedDocument:
    """Raw document produced by the data ingestion component."""

    doc_id: str
    source_path: str
    text: str
    metadata: dict[str, object]

@dataclass(frozen=True)
class DataIngestionArtifact:
    """Final output of the data ingestion component."""

    documents: list[IngestedDocument]
    total_documents: int

@dataclass(frozen=True)
class DocumentChunk:
    """Deterministic text chunk produced from the ingested document."""

    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, object]

@dataclass(frozen=True)
class DataTransformationArtifact:
    """Final output of the data transformation component."""

    chunks: list[DocumentChunk]
    total_chunks: int