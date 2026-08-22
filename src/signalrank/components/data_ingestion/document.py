from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentElement:
    text: str
    element_type: str
    element_index: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    source_path: str
    file_type: str
    elements: tuple[DocumentElement, ...]
    metadata: dict[str, object] = field(default_factory=dict)
