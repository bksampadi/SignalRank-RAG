import os
import sys
from box.exceptions import BoxValueError
from box import ConfigBox
import yaml

from ensure import ensure_annotations
import hashlib

from pathlib import (
    Path,
    PurePosixPath,
)

from signalrank.logging.logger import logging
from signalrank.exception.exception import SignalRankException

@ensure_annotations
def read_text_file(
        file_path: Path,
        encoding: str = "utf-8",
) -> str:
    return file_path.read_text(
        encoding=encoding,
    )

@ensure_annotations
def create_document_id(
        source_reference: str,
        text: str,
        ) -> str:
    """
    Create a reproducible document ID from its relative source path
    and normalised content
    """
    normalised_text = text.replace("\r\n", "\n").replace("\r", "\n")
    identity = f"{source_reference}\0{normalised_text}"

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return f"doc_{digest}"


def get_file_metadata(
        source_reference: str,
        text: str,
        ) -> dict[str, object]:

    source = PurePosixPath(source_reference)

    return {
        "source": source.as_posix(),
        "file_name": source.name,
        "file_extension": source.suffix.lower(),
        "char_count": len(text),
        "word_count": len(text.split()),
    }

def read_yaml(path_to_yaml: Path) -> ConfigBox:

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise SignalRankException (e, sys) from e
