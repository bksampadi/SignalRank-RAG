from importlib.resources import files
from typing import Any
from pathlib import Path

import yaml

class ConfigurationManager:
    """Load SignalRank configuration from a file or packaged defaults."""

    def __init__(
            self,
            config_filepath: str | Path | None = None,
    ):
        if config_filepath is None:
            config_text = (
                files("signalrank.config")
                .joinpath("config.yaml")
                .read_text(encoding="utf-8")
            )
        
        else:
            config_text = Path(config_filepath).read_text(
                encoding="utf-8"
            )

        self.config: dict[str, Any] = yaml.safe_load(config_text)

    def get_config(self) -> dict[str, Any]:
        return self.config