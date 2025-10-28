"""
High-level utilities for exporting chunks to vector databases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Type

from src.workflows.io.reader import read_chunks_jsonl
from src.workflows.io.schema import Chunk
from src.core.vector.base import VectorStore, VectorStoreConfig
from src.core.vector.embeddings import embed_chunks

LOGGER = logging.getLogger(__name__)


def get_vector_store(
    provider: str,
    config: VectorStoreConfig,
) -> VectorStore:
    """
    Get a vector store instance for the specified provider.

    Args:
        provider: Vector database provider ("qdrant", "pinecone", "weaviate")
        config: Vector store configuration

    Returns:
        VectorStore instance

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider.lower()

    if provider == "qdrant":
        from src.core.vector.qdrant_store import QdrantStore
        return QdrantStore(config)
    elif provider == "pinecone":
        from src.core.vector.pinecone_store import PineconeStore
        return PineconeStore(config)
    elif provider == "weaviate":
        from src.core.vector.weaviate_store import WeaviateStore
        return WeaviateStore(config)
    else:
        raise ValueError(
            f"Unsupported vector database provider: {provider}. "
            f"Supported providers: qdrant, pinecone, weaviate"
        )


def export_chunks_to_vectordb(
    chunks: List[Chunk],
    provider: str,
    config: VectorStoreConfig,
    generate_embeddings: bool = True,
    embeddings: Optional[List[List[float]]] = None,
) -> dict:
    """
    Export chunks to a vector database.

    Args:
        chunks: List of chunks to export
        provider: Vector database provider ("qdrant", "pinecone", "weaviate")
        config: Vector store configuration
        generate_embeddings: Whether to generate embeddings for chunks
        embeddings: Pre-computed embeddings (if generate_embeddings=False)

    Returns:
        Dictionary with export results

    Example:
        >>> from src.vectordb import VectorStoreConfig, export_chunks_to_vectordb
        >>> config = VectorStoreConfig(
        ...     api_key="your-api-key",
        ...     index_name="my-index",
        ... )
        >>> result = export_chunks_to_vectordb(
        ...     chunks=chunks,
        ...     provider="qdrant",
        ...     config=config,
        ... )
    """
    if not chunks:
        LOGGER.warning("No chunks provided for export")
        return {"count": 0, "status": "no_chunks"}

    LOGGER.info("Exporting %d chunks to %s", len(chunks), provider)

    # Generate embeddings if needed
    if generate_embeddings:
        if embeddings is not None:
            LOGGER.warning("Both generate_embeddings=True and embeddings provided. Using provided embeddings.")
        else:
            LOGGER.info("Generating embeddings for %d chunks", len(chunks))
            chunk_embeddings = embed_chunks(
                chunks,
                embedding_model=config.embedding_model,
                batch_size=config.batch_size,
            )
            embeddings = [ce.embedding for ce in chunk_embeddings]
    else:
        if embeddings is None:
            raise ValueError("Either generate_embeddings=True or embeddings must be provided")
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

    # Get vector store
    store = get_vector_store(provider, config)

    # Export chunks
    with store:
        # Create index if it doesn't exist
        if not store.index_exists():
            LOGGER.info("Creating index '%s'", config.index_name)
            store.create_index()

        # Upsert chunks
        result = store.upsert(chunks, embeddings)

        # Get stats
        stats = store.get_stats()

    LOGGER.info("Export completed: %s", result)

    return {
        "provider": provider,
        "index_name": config.index_name,
        "chunks_exported": result.get("count", len(chunks)),
        "stats": stats,
    }


def export_jsonl_to_vectordb(
    jsonl_path: Path | str,
    provider: str,
    config: VectorStoreConfig,
) -> dict:
    """
    Export chunks from a JSONL file to a vector database.

    Args:
        jsonl_path: Path to JSONL file containing chunks
        provider: Vector database provider
        config: Vector store configuration

    Returns:
        Dictionary with export results

    Example:
        >>> result = export_jsonl_to_vectordb(
        ...     jsonl_path="data/chunks/document.jsonl",
        ...     provider="qdrant",
        ...     config=VectorStoreConfig(index_name="my-index"),
        ... )
    """
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    LOGGER.info("Loading chunks from %s", jsonl_path)

    # Read chunks from JSONL
    chunks = read_chunks_jsonl(jsonl_path)

    LOGGER.info("Loaded %d chunks from %s", len(chunks), jsonl_path)

    # Export to vector database
    return export_chunks_to_vectordb(
        chunks=chunks,
        provider=provider,
        config=config,
    )


def export_directory_to_vectordb(
    directory: Path | str,
    provider: str,
    config: VectorStoreConfig,
    pattern: str = "*.jsonl",
) -> dict:
    """
    Export all chunks from JSONL files in a directory to a vector database.

    Args:
        directory: Path to directory containing JSONL files
        provider: Vector database provider
        config: Vector store configuration
        pattern: Glob pattern for JSONL files

    Returns:
        Dictionary with export results for all files

    Example:
        >>> result = export_directory_to_vectordb(
        ...     directory="data/chunks",
        ...     provider="pinecone",
        ...     config=VectorStoreConfig(
        ...         api_key="your-api-key",
        ...         index_name="my-index",
        ...     ),
        ... )
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all JSONL files
    jsonl_files = list(directory.glob(pattern))

    if not jsonl_files:
        LOGGER.warning("No JSONL files found in %s with pattern %s", directory, pattern)
        return {
            "provider": provider,
            "index_name": config.index_name,
            "files_processed": 0,
            "total_chunks": 0,
        }

    LOGGER.info("Found %d JSONL files in %s", len(jsonl_files), directory)

    # Load all chunks
    all_chunks = []
    for jsonl_file in jsonl_files:
        try:
            chunks = read_chunks_jsonl(jsonl_file)
            all_chunks.extend(chunks)
            LOGGER.info("Loaded %d chunks from %s", len(chunks), jsonl_file.name)
        except Exception as e:
            LOGGER.error("Error reading %s: %s", jsonl_file, e)

    if not all_chunks:
        LOGGER.warning("No chunks loaded from directory")
        return {
            "provider": provider,
            "index_name": config.index_name,
            "files_processed": len(jsonl_files),
            "total_chunks": 0,
        }

    LOGGER.info("Total chunks loaded: %d", len(all_chunks))

    # Export to vector database
    result = export_chunks_to_vectordb(
        chunks=all_chunks,
        provider=provider,
        config=config,
    )

    result["files_processed"] = len(jsonl_files)
    result["total_chunks"] = len(all_chunks)

    return result
