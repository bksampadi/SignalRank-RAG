from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError

from signalrank.exception.exception import SignalRankException


def create_document_id(
        source_reference: str,
        text: str,
) -> str:
    """
    Create a reproducible document ID from its relative source path
    and normalised content
    """
    normalised_text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    identity = f"{source_reference}\0{normalised_text}"

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return f"doc_{digest}"


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML configuration file."""

    try:
        with path_to_yaml.open(
            "r",
            encoding="utf-8",
        ) as yaml_file:
            content = yaml.safe_load(yaml_file)

        if content is None:
            raise ValueError(
                f"YAML file is empty: {path_to_yaml}"
            )
        
        return ConfigBox(content)
    
    except BoxValueError as exc:
        raise ValueError(
            f"Invalid yaml configuration: {path_to_yaml}"
        ) from exc
    
    except Exception as e:
        raise SignalRankException (e, sys) from e
