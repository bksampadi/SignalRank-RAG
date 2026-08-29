from pathlib import Path

import yaml

from signalrank.components.vector_store.qdrant import (
    create_configured_qdrant_client,
)
from signalrank.config.configuration import ConfigurationManager
from signalrank.constants import CONFIG_FILE_PATH
from signalrank.pipelines.indexing_pipeline import IndexingPipeline


def test_qdrant_index_matches_current_chunks(
    tmp_path: Path,
) -> None:
    with CONFIG_FILE_PATH.open(
        encoding="utf-8",
    ) as file:
        config_data = yaml.safe_load(file)

    config_data["qdrant"]["mode"] = "local"
    config_data["qdrant"]["path"] = str(tmp_path / "qdrant")
    config_data["qdrant"]["recreate_collection"] = True
    config_data["qdrant"]["collection_name"] = "test_index"

    test_config_path = tmp_path / "config.yaml"

    with test_config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config_data,
            file,
        )

    chunks = IndexingPipeline(
        config_filepath=test_config_path,
    ).run()

    expected_ids = {chunk.chunk_id for chunk in chunks}

    config = ConfigurationManager(
        test_config_path,
    ).load()

    client = create_configured_qdrant_client(
        config.qdrant,
    )

    try:
        points, _ = client.scroll(
            collection_name=config.qdrant.collection_name,
            limit=1000,
            with_payload=True,
        )

        qdrant_ids = {
            str(point.payload["chunk_id"])
            for point in points
            if point.payload is not None and "chunk_id" in point.payload
        }

    finally:
        client.close()

    assert expected_ids == qdrant_ids, (
        f"Missing from Qdrant: "
        f"{expected_ids - qdrant_ids}; "
        f"Unexpected in Qdrant: "
        f"{qdrant_ids - expected_ids}"
    )
