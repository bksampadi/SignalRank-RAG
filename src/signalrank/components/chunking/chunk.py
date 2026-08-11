from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    """
    A retrieval-ready chunk derived from a parsed document.
    """

    chunk_id: str
    doc_id: str
    source_path: str
    file_type: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    element_indices: tuple[int, ...]
    element_types: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)