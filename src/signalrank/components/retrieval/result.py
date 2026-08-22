from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    source_path: str
    metadata: dict[str, object] = field(default_factory=dict)
