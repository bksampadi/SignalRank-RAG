from signalrank.components.data_ingestion.finewiki import FineWikiSource


source = FineWikiSource()

for document in source.stream_documents(limit=10):
    print(document.doc_id, document.metadata["title"])