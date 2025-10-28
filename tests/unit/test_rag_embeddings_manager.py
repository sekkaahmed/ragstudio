"""
Tests unitaires pour le module RAG EmbeddingsManager.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.rag.embeddings import (
    EmbeddingsManager,
    get_embeddings_manager,
    EmbeddingModelType,
)


class TestEmbeddingsManager:
    """Tests pour EmbeddingsManager."""

    def test_init_with_recommended_model(self):
        """Test initialisation avec modèle recommandé."""
        manager = EmbeddingsManager(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        assert manager.model_name == "all-MiniLM-L6-v2"
        assert manager.full_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert manager.dimensions == 384
        assert manager.device == "cpu"
        assert manager.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS

    def test_init_with_custom_model(self):
        """Test initialisation avec modèle personnalisé."""
        manager = EmbeddingsManager(
            model_name="custom/model",
            device="cpu",
        )

        assert manager.model_name == "custom/model"
        assert manager.full_model_name == "custom/model"
        assert manager.dimensions is None

    def test_lazy_loading(self):
        """Test que le modèle n'est pas chargé à l'initialisation."""
        manager = EmbeddingsManager(model_name="all-MiniLM-L6-v2")

        assert manager._embeddings is None

        # Accès au modèle déclenche le chargement
        with patch('src.core.rag.embeddings.HuggingFaceEmbeddings') as mock_hf:
            mock_hf.return_value = Mock()
            _ = manager.embeddings
            mock_hf.assert_called_once()

    @pytest.mark.slow
    def test_embed_query_real(self):
        """Test génération d'embedding réel pour une query."""
        manager = EmbeddingsManager(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        embedding = manager.embed_query("Test query")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

        # Vérifier normalisation
        import math
        norm = math.sqrt(sum(x**2 for x in embedding))
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.slow
    def test_embed_documents_real(self):
        """Test génération d'embeddings pour plusieurs documents."""
        manager = EmbeddingsManager(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        texts = [
            "Premier document",
            "Deuxième document",
            "Troisième document",
        ]

        embeddings = manager.embed_documents(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert embeddings[0] != embeddings[1]

    def test_get_model_info(self):
        """Test récupération des infos du modèle."""
        manager = EmbeddingsManager(model_name="all-MiniLM-L6-v2")

        info = manager.get_model_info()

        # Note: 'model_name' is overwritten by RECOMMENDED_MODELS.update()
        assert info['model_name'] == "sentence-transformers/all-MiniLM-L6-v2"
        assert info['full_model_name'] == "sentence-transformers/all-MiniLM-L6-v2"
        assert info['model_type'] == "sentence_transformers"
        assert info['dimensions'] == 384
        assert info['description'] == "Fast, good for most use cases"
        assert info['size_mb'] == 80

    def test_list_recommended_models(self):
        """Test liste des modèles recommandés."""
        models = EmbeddingsManager.list_recommended_models()

        assert "all-MiniLM-L6-v2" in models
        assert "all-mpnet-base-v2" in models
        assert "paraphrase-multilingual-mpnet-base-v2" in models

        model_info = models["all-MiniLM-L6-v2"]
        assert "dimensions" in model_info
        assert "description" in model_info
        assert "size_mb" in model_info

    @patch('src.core.rag.embeddings.HuggingFaceEmbeddings')
    def test_create_sentence_transformers_embeddings(self, mock_hf_embeddings):
        """Test création des embeddings Sentence Transformers."""
        mock_hf_embeddings.return_value = Mock()

        manager = EmbeddingsManager(
            model_name="all-MiniLM-L6-v2",
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        )

        _ = manager.embeddings

        mock_hf_embeddings.assert_called_once()
        call_kwargs = mock_hf_embeddings.call_args.kwargs

        assert call_kwargs['model_name'] == "sentence-transformers/all-MiniLM-L6-v2"
        assert call_kwargs['model_kwargs']['device'] == "cpu"
        assert call_kwargs['encode_kwargs']['normalize_embeddings'] is True

    def test_get_embeddings_manager_singleton(self):
        """Test pattern singleton."""
        import src.core.rag.embeddings as emb_module
        emb_module._embeddings_manager_instance = None

        manager1 = get_embeddings_manager("all-MiniLM-L6-v2")
        manager2 = get_embeddings_manager("all-MiniLM-L6-v2")

        assert manager1 is manager2

        emb_module._embeddings_manager_instance = None


class TestEmbeddingsIntegration:
    """Tests d'intégration pour embeddings."""

    @pytest.mark.slow
    def test_semantic_similarity(self):
        """Test que les embeddings capturent la similarité sémantique."""
        manager = EmbeddingsManager(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        emb1 = manager.embed_query("La grammaire française est complexe")
        emb2 = manager.embed_query("Le français a une grammaire compliquée")
        emb3 = manager.embed_query("J'aime manger des pommes")

        def cosine_similarity(a, b):
            return sum(x * y for x, y in zip(a, b))

        sim_12 = cosine_similarity(emb1, emb2)
        sim_13 = cosine_similarity(emb1, emb3)

        assert sim_12 > sim_13
        assert sim_12 > 0.5

    @pytest.mark.slow
    def test_batch_vs_single(self):
        """Test cohérence batch vs single."""
        manager = EmbeddingsManager(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        text = "Test de cohérence"

        emb_single = manager.embed_query(text)
        emb_batch = manager.embed_documents([text])[0]

        import math
        diff = math.sqrt(sum((a - b)**2 for a, b in zip(emb_single, emb_batch)))
        assert diff < 0.01
