from signalrank.components.chunking.chunking import DocumentChunker
from signalrank.config.configuration import ConfigurationManager
from signalrank.constants import BENCHMARK_CONFIG_FILE_PATH
from signalrank.pipelines.data_ingestion_pipeline import DataIngestionPipeline


def test_chunk_ids_are_deterministic() -> None:
    config = ConfigurationManager(BENCHMARK_CONFIG_FILE_PATH).load()

    documents_1 = DataIngestionPipeline(
        config_filepath=BENCHMARK_CONFIG_FILE_PATH,
    ).run()

    documents_2 = DataIngestionPipeline(
        config_filepath=BENCHMARK_CONFIG_FILE_PATH,
    ).run()

    chunker = DocumentChunker(
        config=config.chunking,
    )

    chunks_1 = chunker.chunk_documents(documents_1)
    chunks_2 = chunker.chunk_documents(documents_2)

    ids_1 = {chunk.chunk_id for chunk in chunks_1}
    ids_2 = {chunk.chunk_id for chunk in chunks_2}

    assert ids_1 == ids_2
