import hashlib
from pathlib import Path

import logfire
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError


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

    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

    return f"doc_{digest}"


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML configuration file."""

    with logfire.span(
        "Read YAML configuration",
        file_path=str(path_to_yaml),
    ):
        try:
            with path_to_yaml.open(
                "r",
                encoding="utf-8",
            ) as yaml_file:
                content = yaml.safe_load(yaml_file)

            if content is None:
                logfire.error(
                    "YAML configuration is empty",
                    file_path=str(path_to_yaml),
                )
                raise ValueError(f"YAML file is empty: {path_to_yaml}")

            return ConfigBox(content)

        except yaml.YAMLError as exc:
            logfire.error(
                "Invalid YAML configuration",
                file_path=str(path_to_yaml),
            )
            raise ValueError(f"Invalid YAML configuration: {path_to_yaml}") from exc

        except BoxValueError as exc:
            logfire.error(
                "Invalid YAML configuration structure",
                file_path=str(path_to_yaml),
            )
            raise ValueError(f"Invalid yaml configuration: {path_to_yaml}") from exc

        except Exception:
            logfire.exception(
                "Failed to read YAML configuration",
                file_path=str(path_to_yaml),
            )
            raise
