from __future__ import annotations

from pathlib import Path

import logfire
from bs4 import BeautifulSoup
from bs4.element import Tag

from signalrank.components.data_ingestion.document import DocumentElement


_BLOCK_TAGS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "blockquote",
    "pre",
    "table",
)


def load_html(
        file_path: Path,
        encoding: str = "utf-8",
) -> list[DocumentElement]:
    """
    Extract meaningful block-level text from an HTML document.
    """

    with logfire.span(
        "Load HTML",
        file_path=str(file_path),
        encoding=encoding,
    ) as span:

        html = file_path.read_text(
            encoding=encoding,
            errors="replace",
        )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "template",
                "svg",
            ]
        ):
            tag.decompose()

        elements: list[DocumentElement] = []

        for block in soup.find_all(_BLOCK_TAGS):
            if not isinstance(block, Tag):
                continue

            # Avoid duplicating table contents:
            # <table> is serialized as one element below.

            if block.name != "table" and block.find_parent("table"):
                continue

            if block.name == "table":
                text = _extract_table_text(block)

            else:
                text = block.get_text(
                    " ",
                    strip=True,
                )

            if not text:
                continue

            elements.append(
                DocumentElement(
                    text=text,
                    element_type=_element_type(block),
                    element_index=len(elements),
                    metadata={
                        "html_tag": block.name,
                    },
                )
            )

        span.set_attribute(
            "elements_extracted",
            len(elements),
        )

        return elements


def _element_type(tag:Tag) -> str:
    """
    Map an HTML tag to a normalized element type.
    """

    if tag.name in {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }:
        return "heading"

    if tag.name == "table":
        return "table"

    if tag.name == "li":
        return "list_item"

    if tag.name == "blockquote":
        return "blockquote"

    if tag.name == "pre":
        return "preformatted"

    return "paragraph"


def _extract_table_text(table: Tag) -> str:
    """
    Serialize an HTML table into deterministic tab-seperated text.
    """

    rows: list[str] = []

    for row in table.find_all["tr"]:
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(
                ["th", "td"],
                recursive=False,
            )
        ]

        cells = [
            cell
            for cell in cells
            if cell
        ]

        if cells:
            row.append("\t".join(cells))

    return "\n".join(rows)