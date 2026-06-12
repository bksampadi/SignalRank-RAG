from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format= '[%(asctime)s]: %(message)s:')


project_name="signalrank"

list_of_files=[
    ".github/workflows/.gitkeep",

    # Package root
    f"src/{project_name}/__init__.py",

    # Core infrastructure
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/logging/logger.py",

    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/exception/exception.py",

    f"src/{project_name}/constants/__init__.py",

    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",

    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/chunking.py",
    f"src/{project_name}/components/embedding.py",
    f"src/{project_name}/components/vector_store.py",
    f"src/{project_name}/components/retrieval.py",
    f"src/{project_name}/components/reranking.py",
    f"src/{project_name}/components/llm_generator.py",
    f"src/{project_name}/components/prompt_builder.py",

    # Core
    f"src/{project_name}/core/__init__.py",

    # Pipelines
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/inference_pipeline.py",

    # Entity
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/artifact_entity.py",
    f"src/{project_name}/entity/config_entity.py",

    # Schema
    f"src/{project_name}/schema/__init__.py",
    f"src/{project_name}/schema/document_schema.py",

    # Evaluation
    f"src/{project_name}/evaluation/__init__.py",
    f"src/{project_name}/evaluation/retrieval_eval.py",
    f"src/{project_name}/evaluation/rag_eval.py",

    # Prompts
    f"src/{project_name}/prompts/__init__.py",
    f"src/{project_name}/prompts/rag_prompts.py",

    # Utils
    f"src/{project_name}/utils/__init__.py",

    # Documentation
    "docs/.gitkeep",

    # Configs
    "configs/config.yaml",
    "params.yaml",

    # Entry points
    "app.py",
    "main.py",

    # Packaging
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "Dockerfile",

    # Tests
    "tests/__init__.py",
    "tests/test_exception.py",
    "tests/test_configuration.py",
    "tests/test_logger.py",
    "tests/test_data_ingestion.py",
    "tests/test_chunking.py",
    "tests/test_embedding.py",
    "tests/test_vector_store.py",
    "tests/test_retrieval.py",
    "tests/test_reranking.py",

    # Research
    "research/research.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent
    filename = filepath.name

    filedir.mkdir(parents=True, exist_ok=True)
    logging.info(
            f"Creating directory: {filedir} for file: {filename}"
            )
    
    if (not filepath.exists()) or (filepath.stat().st_size == 0):
        filepath.touch()
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")