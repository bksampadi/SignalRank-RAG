from dataclasses import dataclass


@dataclass
class DocumentRecord:
    content: str

    source: str
    source_type: str

    page_number: int | None = None

    document_id: str | None = None

    author: str | None = None
    created_at: str | None = None

    chunk_id: str | None = None
