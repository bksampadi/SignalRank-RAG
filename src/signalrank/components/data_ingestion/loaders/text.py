from __future__ import annotations

from pathlib import Path

import logfire

from signalrank.components.data_ingestion.document import DocumentElement


def load_text(
        file_path: Path,
        *,
        encoding: str = "utf-8",
) -> List[DocumentElement]:
    """
    Load a plain-text or Markdown file.
    """

    with logfire.span(
        "Load text document",
        file_path=str(file_path),
        encoding=encoding,
    ) as span:

        text = file_path.read_text(
            encoding=encoding,
            errors="replace",
        ).strip()

        if not text:
            span.set_attribute("text_extacted", False)
            return []

        span.set_attribute("text_extracted", True)
        span.set_attribute("character_count", len(text))

        return [
            DocumentElement(
                text=text,
                element_type="text",
                element_index=0,
                metadata={
                    "extension": file_path.suffix.lower(),
                },
            )
        ]