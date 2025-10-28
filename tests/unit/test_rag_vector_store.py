"""
Tests unitaires pour le module RAG VectorStoreManager.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.core.rag.vector_store import VectorStoreManager, create_vector_store
from src.core.rag.embeddings import EmbeddingsManager


@pytest.fixture
def mock_embeddings_manager():
    """Mock EmbeddingsManager pour les tests."""
    manager = Mock(spec=EmbeddingsManager)
    manager.model_name = "test-model"
    manager.embeddings = Mock()
    return manager


@pytest.fixture
def mock_metadata_store():
    """Mock MetadataStore pour les tests."""
    store = Mock()
    store.update_chunk_vector_id = Mock()
    return store


class TestVectorStoreManager:
    """Tests pour VectorStoreManager."""

    def test_init_local(self, mock_embeddings_manager):
        """Test initialisation avec Qdrant local."""
        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            collection_name="test_collection",
            use_local=True,
        )

        assert manager.collection_name == "test_collection"
        assert manager.use_local is True
        assert manager._vector_store is None

    def test_init_remote(self, mock_embeddings_manager):
        """Test initialisation avec Qdrant distant."""
        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            collection_name="test_collection",
            use_local=False,
            qdrant_url="http://localhost:6333",
        )

        assert manager.use_local is False
        assert manager.qdrant_url == "http://localhost:6333"

    def test_lazy_loading_qdrant_client(self, mock_embeddings_manager):
        """Test chargement lazy du client Qdrant."""
        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            use_local=True,
        )

        assert manager._qdrant_client is None

        with patch('src.core.rag.vector_store.QdrantClient') as mock_qdrant:
            _ = manager.qdrant_client
            mock_qdrant.assert_called_once_with(location=":memory:")

    def test_lazy_loading_vector_store(self, mock_embeddings_manager):
        """Test chargement lazy du vector store."""
        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            use_local=True,
        )

        assert manager._vector_store is None

        with patch('src.core.rag.vector_store.Qdrant') as mock_qdrant_store:
            _ = manager.vector_store
            mock_qdrant_store.assert_called_once()

    @patch('src.core.rag.vector_store.Qdrant')
    @patch('src.core.rag.vector_store.QdrantClient')
    def test_add_documents_without_metadata_store(
        self, mock_client, mock_qdrant, mock_embeddings_manager
    ):
        """Test ajout de documents sans metadata store."""
        mock_vector_store = MagicMock()
        mock_vector_store.add_documents.return_value = ['id1', 'id2']
        mock_qdrant.return_value = mock_vector_store

        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            use_local=True,
        )

        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
        ]

        vector_ids = manager.add_documents(docs)

        assert vector_ids == ['id1', 'id2']
        mock_vector_store.add_documents.assert_called_once_with(docs)

    @patch('src.core.rag.vector_store.Qdrant')
    @patch('src.core.rag.vector_store.QdrantClient')
    def test_add_documents_with_metadata_store(
        self, mock_client, mock_qdrant, mock_embeddings_manager, mock_metadata_store
    ):
        """Test ajout de documents avec sync metadata store."""
        mock_vector_store = MagicMock()
        mock_vector_store.add_documents.return_value = ['vec1', 'vec2']
        mock_qdrant.return_value = mock_vector_store

        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            metadata_store=mock_metadata_store,
            use_local=True,
        )

        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
        ]
        chunk_ids = ['chunk1', 'chunk2']

        vector_ids = manager.add_documents(docs, chunk_ids=chunk_ids)

        assert vector_ids == ['vec1', 'vec2']
        assert mock_metadata_store.update_chunk_vector_id.call_count == 2

    @patch('src.core.rag.vector_store.Qdrant')
    @patch('src.core.rag.vector_store.QdrantClient')
    def test_similarity_search(
        self, mock_client, mock_qdrant, mock_embeddings_manager
    ):
        """Test recherche par similarité."""
        mock_vector_store = MagicMock()
        mock_results = [
            Document(page_content="Result 1", metadata={}),
            Document(page_content="Result 2", metadata={}),
        ]
        mock_vector_store.similarity_search.return_value = mock_results
        mock_qdrant.return_value = mock_vector_store

        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            use_local=True,
        )

        results = manager.similarity_search("test query", k=2)

        assert len(results) == 2
        assert results == mock_results
        mock_vector_store.similarity_search.assert_called_once_with("test query", k=2)

    @patch('src.core.rag.vector_store.Qdrant')
    @patch('src.core.rag.vector_store.QdrantClient')
    def test_similarity_search_with_score(
        self, mock_client, mock_qdrant, mock_embeddings_manager
    ):
        """Test recherche avec scores."""
        mock_vector_store = MagicMock()
        mock_results = [
            (Document(page_content="Result 1", metadata={}), 0.95),
            (Document(page_content="Result 2", metadata={}), 0.85),
        ]
        mock_vector_store.similarity_search_with_score.return_value = mock_results
        mock_qdrant.return_value = mock_vector_store

        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            use_local=True,
        )

        results = manager.similarity_search_with_score("test query", k=2)

        assert len(results) == 2
        assert results[0][1] == 0.95
        assert results[1][1] == 0.85

    @patch('src.core.rag.vector_store.QdrantClient')
    def test_get_collection_info_local(self, mock_client, mock_embeddings_manager):
        """Test récupération des infos de collection (local)."""
        mock_qdrant_client = MagicMock()
        mock_client.return_value = mock_qdrant_client

        manager = VectorStoreManager(
            embeddings_manager=mock_embeddings_manager,
            collection_name="test_collection",
            use_local=True,
        )

        # Mock in-memory count
        with patch('src.core.rag.vector_store.Qdrant') as mock_qdrant_store:
            mock_vs = MagicMock()
            mock_qdrant_store.return_value = mock_vs
            _ = manager.vector_store

            info = manager.get_collection_info()

            assert info['collection_name'] == "test_collection"
            assert info['storage_type'] == "in-memory"

    def test_create_vector_store_helper(self, mock_embeddings_manager):
        """Test helper function create_vector_store."""
        manager = create_vector_store(
            embeddings_manager=mock_embeddings_manager,
            collection_name="test",
            use_local=True,
        )

        assert isinstance(manager, VectorStoreManager)
        assert manager.collection_name == "test"


class TestVectorStoreIntegration:
    """Tests d'intégration pour VectorStoreManager."""

    @pytest.mark.slow
    def test_add_and_search_real(self):
        """Test ajout et recherche réels avec embeddings."""
        from src.core.rag.embeddings import get_embeddings_manager

        # Reset singleton
        import src.core.rag.embeddings as emb_module
        emb_module._embeddings_manager_instance = None

        embeddings_manager = get_embeddings_manager(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        manager = create_vector_store(
            embeddings_manager=embeddings_manager,
            collection_name="test_integration",
            use_local=True,
        )

        docs = [
            Document(
                page_content="La grammaire française est complexe",
                metadata={'source': 'doc1'}
            ),
            Document(
                page_content="Python est un langage de programmation",
                metadata={'source': 'doc2'}
            ),
            Document(
                page_content="Le français a des règles grammaticales",
                metadata={'source': 'doc3'}
            ),
        ]

        vector_ids = manager.add_documents(docs)
        assert len(vector_ids) == 3

        results = manager.similarity_search("grammaire français", k=2)
        assert len(results) <= 2

        # Les résultats devraient être doc1 et doc3 (plus pertinents)
        result_sources = [r.metadata['source'] for r in results]
        assert 'doc1' in result_sources or 'doc3' in result_sources

        # Cleanup
        emb_module._embeddings_manager_instance = None

    @pytest.mark.slow
    def test_metadata_store_sync(self):
        """Test synchronisation avec metadata store."""
        from src.core.rag.embeddings import get_embeddings_manager

        import src.core.rag.embeddings as emb_module
        emb_module._embeddings_manager_instance = None

        embeddings_manager = get_embeddings_manager(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        mock_metadata_store = Mock()
        mock_metadata_store.update_chunk_vector_id = Mock()

        manager = create_vector_store(
            embeddings_manager=embeddings_manager,
            metadata_store=mock_metadata_store,
            collection_name="test_sync",
            use_local=True,
        )

        docs = [
            Document(page_content="Test 1", metadata={}),
            Document(page_content="Test 2", metadata={}),
        ]
        chunk_ids = ['chunk1', 'chunk2']

        manager.add_documents(docs, chunk_ids=chunk_ids)

        # Vérifier que les chunk_ids ont été synchronisés
        assert mock_metadata_store.update_chunk_vector_id.call_count == 2

        emb_module._embeddings_manager_instance = None
