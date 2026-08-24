from argparse import ArgumentParser
from pathlib import Path

from signalrank.constants import CONFIG_FILE_PATH
from signalrank.pipelines.indexing_pipeline import IndexingPipeline


def parse_args():
    parser = ArgumentParser(
        description="Index a corpus using a SignalRank configuration."
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the configured Qdrant collection.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE_PATH,
        help="Path to the configuration YAML file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = IndexingPipeline(
        config_filepath=args.config,
    ).run(
        recreate=args.recreate,
    )

    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
