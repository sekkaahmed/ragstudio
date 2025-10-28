"""
Integration tests for vector database connectors.
"""

import pytest

from src.workflows.io.schema import Document, make_chunks
from src.core.vector.base import VectorStoreConfig
from src.core.vector.embeddings import embed_chunks
from src.core.vector.exporter import get_vector_store, export_chunks_to_vectordb

# Check if Qdrant is available
try:
    import qdrant_client  # noqa: F401
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    doc = Document(
        text="Sample document text",
        source_path="test.txt",
        language="en",
    )

    text_chunks = [
        "This is the first chunk about machine learning.",
        "This is the second chunk about natural language processing.",
        "This is the third chunk about vector databases.",
    ]

    return make_chunks(doc, text_chunks)


@pytest.fixture
def vector_config():
    """Create vector store configuration for testing."""
    return VectorStoreConfig(
        index_name="test-index",
        embedding_dimension=384,
        batch_size=10,
    )


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_qdrant_in_memory_connection(vector_config):
    """Test connecting to in-memory Qdrant."""
    from src.core.vector.qdrant_store import QdrantStore

    store = QdrantStore(vector_config)
    store.connect()

    assert store.client is not None
    store.close()


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_qdrant_create_index(vector_config):
    """Test creating a Qdrant collection."""
    from src.core.vector.qdrant_store import QdrantStore

    store = QdrantStore(vector_config)

    with store:
        # Create index
        store.create_index()

        # Check if index exists
        assert store.index_exists()

        # Get stats
        stats = store.get_stats()
        assert stats["index_name"] == "test-index"
        assert stats["connected"] is True


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_qdrant_upsert_and_search(vector_config, sample_chunks):
    """Test upserting chunks and searching in Qdrant."""
    from src.core.vector.qdrant_store import QdrantStore

    store = QdrantStore(vector_config)

    with store:
        # Generate embeddings
        chunk_embeddings = embed_chunks(sample_chunks, batch_size=3)
        embeddings = [ce.embedding for ce in chunk_embeddings]

        # Upsert chunks
        result = store.upsert(sample_chunks, embeddings)

        assert result["count"] == 3
        assert result["collection"] == "test-index"

        # Search with first embedding
        query_embedding = embeddings[0]
        results = store.search(query_embedding, top_k=2)

        assert len(results) <= 2
        assert results[0]["id"] == sample_chunks[0].id  # First should match itself
        assert results[0]["score"] > 0.9  # High similarity


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_qdrant_delete(vector_config, sample_chunks):
    """Test deleting vectors from Qdrant."""
    from src.core.vector.qdrant_store import QdrantStore

    store = QdrantStore(vector_config)

    with store:
        # Generate and upsert
        chunk_embeddings = embed_chunks(sample_chunks, batch_size=3)
        embeddings = [ce.embedding for ce in chunk_embeddings]
        store.upsert(sample_chunks, embeddings)

        # Delete by ID
        delete_result = store.delete(ids=[sample_chunks[0].id])
        assert delete_result["count"] == 1

        # Search should not return deleted chunk
        results = store.search(embeddings[0], top_k=3)
        assert all(r["id"] != sample_chunks[0].id for r in results)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_qdrant_filter_search(vector_config, sample_chunks):
    """Test filtered search in Qdrant."""
    from src.core.vector.qdrant_store import QdrantStore

    # Add document_id to metadata for filtering
    for chunk in sample_chunks:
        chunk.metadata["document_id"] = chunk.document_id

    store = QdrantStore(vector_config)

    with store:
        # Generate and upsert
        chunk_embeddings = embed_chunks(sample_chunks, batch_size=3)
        embeddings = [ce.embedding for ce in chunk_embeddings]
        store.upsert(sample_chunks, embeddings)

        # Search with filter
        filter_dict = {"document_id": sample_chunks[0].document_id}
        results = store.search(
            embeddings[0],
            top_k=10,
            filter_dict=filter_dict,
        )

        # All results should match the filter
        assert all(r["document_id"] == sample_chunks[0].document_id for r in results)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_get_vector_store_qdrant(vector_config):
    """Test getting Qdrant store via factory function."""
    store = get_vector_store("qdrant", vector_config)

    from src.core.vector.qdrant_store import QdrantStore
    assert isinstance(store, QdrantStore)


def test_get_vector_store_invalid_provider(vector_config):
    """Test getting store with invalid provider."""
    with pytest.raises(ValueError, match="Unsupported vector database provider"):
        get_vector_store("invalid_provider", vector_config)


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_export_chunks_to_vectordb(vector_config, sample_chunks):
    """Test high-level export function."""
    result = export_chunks_to_vectordb(
        chunks=sample_chunks,
        provider="qdrant",
        config=vector_config,
        generate_embeddings=True,
    )

    assert result["provider"] == "qdrant"
    assert result["index_name"] == "test-index"
    assert result["chunks_exported"] == 3
    assert "stats" in result


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_export_with_precomputed_embeddings(vector_config, sample_chunks):
    """Test export with pre-computed embeddings."""
    # Generate embeddings separately
    chunk_embeddings = embed_chunks(sample_chunks, batch_size=3)
    embeddings = [ce.embedding for ce in chunk_embeddings]

    result = export_chunks_to_vectordb(
        chunks=sample_chunks,
        provider="qdrant",
        config=vector_config,
        generate_embeddings=False,
        embeddings=embeddings,
    )

    assert result["chunks_exported"] == 3


def test_export_no_chunks(vector_config):
    """Test export with no chunks."""
    result = export_chunks_to_vectordb(
        chunks=[],
        provider="qdrant",
        config=vector_config,
    )

    assert result["count"] == 0
    assert result["status"] == "no_chunks"


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant client not installed")
def test_context_manager(vector_config):
    """Test vector store context manager."""
    from src.core.vector.qdrant_store import QdrantStore

    store = QdrantStore(vector_config)

    # Before context: not connected
    assert store.client is None

    with store:
        # Inside context: connected
        assert store.client is not None

        # Create index works
        store.create_index()
        assert store.index_exists()

    # After context: closed
    assert store.client is None
