from pathlib import Path

import logfire

from signalrank.components.ranking.base import Reranker
from signalrank.components.ranking.flashrank import FlashRankReranker
from signalrank.config.configuration import ConfigurationManager
from signalrank.constants import CONFIG_FILE_PATH


class RankingPipeline:
    """Build the configured ranking stack."""

    def __init__(
        self,
        config_filepath: str | Path = CONFIG_FILE_PATH,
    ):
        self._config_filepath = Path(config_filepath)

    def build(self) -> dict[str, Reranker]:
        config = ConfigurationManager(self._config_filepath).load()

        with logfire.span(
            "Build ranking pipeline",
            ranking_provider=config.ranking.provider,
            ranking_model=config.ranking.model_name,
        ):
            if config.ranking.provider == "flashrank":
                reranker = FlashRankReranker(
                    model_name=config.ranking.model_name,
                    max_length=config.ranking.max_length,
                )

                return {
                    "flashrank": reranker,
                }

            raise ValueError(f"Unsupported ranking provider:{config.ranking.provider}")
