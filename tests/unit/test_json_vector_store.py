"""
Tests unitaires pour JSONVectorStore.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.core.rag.json_vector_store import JSONVectorStore


@pytest.fixture
def temp_dir():
    """Temporary directory for tests."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_embeddings():
    """Mock embeddings with proper methods."""
    embeddings = Mock()
    # embed_documents returns list of embeddings - side_effect for multiple calls
    embeddings.embed_documents = Mock(side_effect=lambda texts: [
        [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] for i in range(len(texts))
    ])
    # embed_query returns single embedding
    embeddings.embed_query = Mock(return_value=[0.2, 0.3, 0.4])
    return embeddings


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(page_content="This is document 1", metadata={"id": 1, "source": "test"}),
        Document(page_content="This is document 2", metadata={"id": 2, "source": "test"}),
        Document(page_content="This is document 3", metadata={"id": 3, "source": "test"}),
    ]


class TestJSONVectorStoreInit:
    """Tests for JSONVectorStore initialization."""

    def test_init_basic(self, mock_embeddings):
        """Test basic initialization."""
        store = JSONVectorStore(embedding=mock_embeddings)

        assert store._embedding == mock_embeddings
        assert store.persist_directory is None
        assert store.collection_name == "default"
        assert store._documents == []
        assert store._embeddings is None
        assert store._document_ids == []

    def test_init_with_persist_dir(self, mock_embeddings, temp_dir):
        """Test initialization with persist directory."""
        store = JSONVectorStore(
            embedding=mock_embeddings,
            persist_directory=str(temp_dir)
        )

        assert store.persist_directory == temp_dir
        assert store.persist_directory.exists()

    def test_init_with_collection_name(self, mock_embeddings):
        """Test initialization with custom collection name."""
        store = JSONVectorStore(
            embedding=mock_embeddings,
            collection_name="test_collection"
        )

        assert store.collection_name == "test_collection"


class TestJSONVectorStoreAddDocuments:
    """Tests for adding documents to the store."""

    def test_add_documents_basic(self, mock_embeddings, sample_documents):
        """Test adding documents."""
        store = JSONVectorStore(embedding=mock_embeddings)

        ids = store.add_documents(sample_documents)

        assert len(ids) == 3
        assert len(store._documents) == 3
        assert len(store._document_ids) == 3
        assert store._embeddings is not None
        assert store._embeddings.shape == (3, 3)

        # Verify embed_documents was called
        mock_embeddings.embed_documents.assert_called_once()
        call_args = mock_embeddings.embed_documents.call_args[0][0]
        assert len(call_args) == 3

    def test_add_documents_incremental(self, mock_embeddings, sample_documents):
        """Test adding documents incrementally."""
        store = JSONVectorStore(embedding=mock_embeddings)

        # Add first batch
        ids1 = store.add_documents(sample_documents[:2])
        assert len(ids1) == 2
        assert len(store._documents) == 2

        # Add second batch
        ids2 = store.add_documents(sample_documents[2:])
        assert len(ids2) == 1
        assert len(store._documents) == 3
        assert store._embeddings.shape == (3, 3)

    def test_add_documents_with_ids(self, mock_embeddings, sample_documents):
        """Test adding documents with custom IDs."""
        store = JSONVectorStore(embedding=mock_embeddings)
        custom_ids = ["id1", "id2", "id3"]

        ids = store.add_documents(sample_documents, ids=custom_ids)

        assert ids == custom_ids
        assert store._document_ids == custom_ids


class TestJSONVectorStoreSimilaritySearch:
    """Tests for similarity search."""

    def test_similarity_search_basic(self, mock_embeddings, sample_documents):
        """Test basic similarity search."""
        store = JSONVectorStore(embedding=mock_embeddings)
        store.add_documents(sample_documents)

        results = store.similarity_search("test query", k=2)

        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

        # Verify embed_query was called
        mock_embeddings.embed_query.assert_called_once_with("test query")

    def test_similarity_search_k_greater_than_docs(self, mock_embeddings, sample_documents):
        """Test search with k > number of documents."""
        store = JSONVectorStore(embedding=mock_embeddings)
        store.add_documents(sample_documents[:2])

        results = store.similarity_search("test query", k=10)

        # Should return all available documents (2)
        assert len(results) == 2

    def test_similarity_search_with_score(self, mock_embeddings, sample_documents):
        """Test similarity search with scores."""
        store = JSONVectorStore(embedding=mock_embeddings)
        store.add_documents(sample_documents)

        results = store.similarity_search_with_score("test query", k=2)

        assert len(results) == 2
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_similarity_search_empty_store(self, mock_embeddings):
        """Test search on empty store."""
        store = JSONVectorStore(embedding=mock_embeddings)

        results = store.similarity_search("test query")

        assert results == []


class TestJSONVectorStorePersistLoad:
    """Tests for persist and load functionality."""

    def test_persist_basic(self, mock_embeddings, sample_documents, temp_dir):
        """Test persisting store to disk."""
        store = JSONVectorStore(
            embedding=mock_embeddings,
            persist_directory=str(temp_dir)
        )
        store.add_documents(sample_documents)

        store.persist()

        # Check files were created
        docs_file = temp_dir / "documents.json"
        embeddings_file = temp_dir / "embeddings.npy"
        index_file = temp_dir / "index.json"

        assert docs_file.exists()
        assert embeddings_file.exists()
        assert index_file.exists()

    def test_persist_without_directory(self, mock_embeddings, sample_documents):
        """Test persist fails without directory."""
        store = JSONVectorStore(embedding=mock_embeddings)
        store.add_documents(sample_documents)

        with pytest.raises(ValueError, match="persist_directory"):
            store.persist()

    def test_load_basic(self, mock_embeddings, sample_documents, temp_dir):
        """Test loading store from disk."""
        # Create and persist store
        store1 = JSONVectorStore(
            embedding=mock_embeddings,
            persist_directory=str(temp_dir)
        )
        store1.add_documents(sample_documents)
        store1.persist()

        # Load into new store
        store2 = JSONVectorStore.load(
            persist_directory=str(temp_dir),
            embedding=mock_embeddings,
            collection_name="default"
        )

        assert len(store2._documents) == 3
        assert len(store2._document_ids) == 3
        assert store2._embeddings is not None
        assert store2._embeddings.shape == (3, 3)

    def test_load_nonexistent_directory(self, mock_embeddings):
        """Test loading from nonexistent directory (should create empty store)."""
        # Load from nonexistent path should succeed but create empty store
        store = JSONVectorStore.load(
            persist_directory="/nonexistent/path",
            embedding=mock_embeddings
        )

        # Store should be empty
        assert len(store._documents) == 0


class TestJSONVectorStoreDelete:
    """Tests for deleting documents."""

    def test_delete_documents(self, mock_embeddings, sample_documents):
        """Test deleting documents by IDs."""
        store = JSONVectorStore(embedding=mock_embeddings)
        ids = store.add_documents(sample_documents)

        # Delete first document
        success = store.delete([ids[0]])

        assert success is True
        assert len(store._documents) == 2
        assert len(store._document_ids) == 2
        assert store._embeddings.shape == (2, 3)
        assert ids[0] not in store._document_ids

    def test_delete_nonexistent_id(self, mock_embeddings, sample_documents):
        """Test deleting nonexistent ID."""
        store = JSONVectorStore(embedding=mock_embeddings)
        store.add_documents(sample_documents)

        # Try to delete nonexistent ID
        success = store.delete(["nonexistent_id"])

        # Should not crash, but may return False or just not affect store
        assert len(store._documents) == 3  # No change


class TestJSONVectorStoreGetDocuments:
    """Tests for retrieving documents."""

    def test_get_all_documents(self, mock_embeddings, sample_documents):
        """Test getting all documents."""
        store = JSONVectorStore(embedding=mock_embeddings)
        store.add_documents(sample_documents)

        docs = store._documents

        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)

    def test_get_document_count(self, mock_embeddings, sample_documents):
        """Test getting document count."""
        store = JSONVectorStore(embedding=mock_embeddings)

        assert len(store._documents) == 0

        store.add_documents(sample_documents)

        assert len(store._documents) == 3
