from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format= '[%(asctime)s]: %(message)s:')


project_name = "signalrank"

PROJECT_STRUCTURE = {
    ".github/workflows": [".gitkeep"],

    # Package root
    f"src/{project_name}": ["__init__.py"],

    # Core infrastructure
    f"src/{project_name}/logging": [
        "__init__.py",
        "logger.py",
    ],

    f"src/{project_name}/exception": [
        "__init__.py",
        "exception.py",
    ],

    f"src/{project_name}/constants": [
        "__init__.py",
    ],

    f"src/{project_name}/config": [
        "__init__.py",
        "configuration.py",
    ],

    f"src/{project_name}/utils": [
        "__init__.py",
        "common.py",
    ],

    # Components
    f"src/{project_name}/components": [
        "__init__.py",
        "data_ingestion.py",
        "data_validation.py",
        "data_transformation.py",
        "chunking.py",
        "embedding.py",
        "vector_store.py",
        "retrieval.py",
        "reranking.py",
        "signal_ledger.py",
        "answerability.py",
        "opposition_retrieval.py",
        "llm_generator.py",
        "prompt_builder.py",
    ],

    # Core
    f"src/{project_name}/core": [
        "__init__.py",
        "evidence_admission.py",
        "signal_trace.py",
    ],

    # Pipelines
    f"src/{project_name}/pipelines": [
        "__init__.py",
        "training_pipeline.py",
        "inference_pipeline.py",
    ],

    # Entity
    f"src/{project_name}/entity": [
        "__init__.py",
        "artifact_entity.py",
        "config_entity.py",
        "signal_entity.py",
    ],

    # Schema
    f"src/{project_name}/schema": [
        "__init__.py",
        "document_schema.py",
    ],

    # Evaluation
    f"src/{project_name}/evaluation": [
        "__init__.py",
        "retrieval_eval.py",
        "rag_eval.py",
    ],

    # Prompts
    f"src/{project_name}/prompts": [
        "__init__.py",
        "rag_prompts.py",
    ],

    # Documentation
    "docs": [
        ".gitkeep",
    ],

    # Configs
    "configs": [
        "config.yaml",
    ],

    
    # Entry point and Packaging
    ".": [
        "params.yaml",
        "app.py",
        "main.py",
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "Dockerfile",
    ],

    # Tests
    "tests": [
        "__init__.py",
        "test_exception.py",
        "test_configuration.py",
        "test_logger.py",
        "test_data_ingestion.py",
        "test_chunking.py",
        "test_embedding.py",
        "test_vector_store.py",
        "test_retrieval.py",
        "test_reranking.py",
        "test_signal_ledger.py",
        "test_answerability.py",
    ],

    # Research
     "research": [
        "research.ipynb",
    ],
}

for directory, files in PROJECT_STRUCTURE.items():
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Ensured directory exists: {directory_path}")

    for file_name in files:
        file_path = directory_path / file_name

        if not file_path.exists() or file_path.stat().st_size == 0:
            file_path.touch()
            logging.info(f"Created empty file: {file_path}")
        else:
            logging.info(f"File already exists: {file_path}")