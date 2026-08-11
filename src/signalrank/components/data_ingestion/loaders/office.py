from __future__ import annotations

from pathlib import Path

import logfire
from unstructured.partition.auto import partition

from signalrank.components.data_ingestion.document import DocumentElement


def load_office(file_path: Path) -> list[DocumentElement]:
    """
    Extract structured elements from DOCX, PPTX, and XLSX documents.
    """

    with logfire.span(
        "Load Office document",
        file_path=str(file_path),
        file_type=file_path.suffix.lower(),
    ) as span:

        raw_elements = partition(
            filename=str(file_path),
        )

        elements: list[DocumentElement] = []

        for raw_element in raw_elements:
            text = str(raw_element).strip()

            if not text:
                continue

            metadata = raw_element.metadata.to_dict()

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