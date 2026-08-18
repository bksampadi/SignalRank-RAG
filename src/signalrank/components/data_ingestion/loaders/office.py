from collections.abc import Callable
from pathlib import Path

import logfire
from unstructured.documents.elements import Element
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx

from signalrank.components.data_ingestion.document import DocumentElement

_PARTITIONERS: dict[str, Callable[..., list[Element]]] = {
    ".docx": partition_docx,
    ".pptx": partition_pptx,
    ".xlsx": partition_xlsx,
}

_ELEMENT_TYPE_MAP = {
    "Title": "heading",
    "NarrativeText": "paragraph",
    "UncategorizedText": "paragraph",
    "ListItem": "list_item",
    "Table": "table",
    "Header": "header",
    "Footer": "footer",
    "FigureCaption": "figure_caption",
    "PageNumber": "page_number",
    "Formula": "formula",
    "CodeSnippet": "code",
    "Image": "image",
    "Address": "address",
    "EmailAddress": "email_address",
}

def load_office(file_path: Path) -> list[DocumentElement]:
    """
    Load an Office document into structured document elements.
    """

    extension = file_path.suffix.lower()

    with logfire.span(
        "Load Office document",
        file_path=str(file_path),
        file_type=extension,
    ) as span:
        try:
            partitioner = _PARTITIONERS[extension]
            
        except KeyError as exc:
            logfire.error(
                "Unsupported Office file type",
                file_path=str(file_path),
                file_type=extension,
            )

            raise ValueError(
                f"Unsupported Office file type: {extension}"
            ) from exc
         
        try:
            raw_elements = partitioner(
                filename=str(file_path),
            )

            elements: list[DocumentElement] = []

            for raw_element in raw_elements:
                text = str(raw_element).strip()

                if not text:
                    continue

                metadata = (
                    raw_element.metadata.to_dict()
                    if raw_element.metadata is not None
                    else {}
                )

                elements.append(
                    DocumentElement(
                        text=text,
                        element_type=_element_type(
                            raw_element.category
                        ),
                        element_index=len(elements),
                        metadata=metadata,
                    )
                )

            span.set_attribute(
                "elements_extracted",
                len(elements),
            )

            return elements

        except Exception:
            logfire.exception(
                "Office document loading failed",
                file_path=str(file_path),
                file_type=extension,
            )
            raise

def _element_type(category: str) -> str:
    """
    Map an Unstructured categroy to a SignalRank element type.
    """

    return _ELEMENT_TYPE_MAP.get(
        category,
        "paragraph",
    )