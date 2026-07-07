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