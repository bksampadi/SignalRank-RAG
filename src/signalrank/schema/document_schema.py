from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentRecord:
    content: str

    source: str
    source_type: str

    page_number: Optional[int] = None

    document_id: Optional[str] = None

    author: Optional[str] = None
    created_at: Optional[str] = None

    chunk_id: Optional[str] = None
