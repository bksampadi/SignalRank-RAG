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


def load_office(file_path: Path) -> list[DocumentElement]:
    """
    Load an Office document into structured document elements.
    """

    extension = file_path.suffix.lower()

    try:
        partitioner = _PARTITIONERS[extension]
    except KeyError as exc:
        raise ValueError(
        f"Unsupported Office file type: {extension}"
        ) from exc

    with logfire.span(
        "Load Office document",
        file_path=str(file_path),
        file_type=extension,
    ) as span:

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
                    element_type=raw_element.category.lower(),
                    element_index=len(elements),
                    metadata=metadata,
                )
            )

        span.set_attribute(
            "elements_extracted",
            len(elements),
        )

        return elements