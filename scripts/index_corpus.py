from signalrank.pipelines.indexing_pipeline import IndexingPipeline


def main() -> None:
    chunks = IndexingPipeline().run()
    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
