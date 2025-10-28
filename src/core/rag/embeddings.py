"""
Embeddings management for Atlas-RAG.

Provides unified interface for different embedding models:
- Sentence Transformers (local, free)
- OpenAI Embeddings (API, paid)
- HuggingFace Embeddings (various models)
"""

from __future__ import annotations

import logging
from typing import List, Optional
from enum import Enum

from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)

LOGGER = logging.getLogger(__name__)


class EmbeddingModelType(Enum):
    """Supported embedding model types."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class EmbeddingsManager:
    """
    Manages embedding models for Atlas-RAG.

    Provides unified interface for different embedding providers
    with caching and batch processing support.
    """

    # Recommended models
    RECOMMENDED_MODELS = {
        # Small & Fast (384 dims)
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Fast, good for most use cases",
            "size_mb": 80,
        },
        # Balanced (768 dims)
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "description": "Best quality/speed tradeoff",
            "size_mb": 420,
        },
        # Multilingual (768 dims)
        "paraphrase-multilingual-mpnet-base-v2": {
            "dimensions": 768,
            "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "description": "Best for French + multilingual",
            "size_mb": 970,
        },
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_type: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS,
        device: str = "cpu",
        cache_folder: Optional[str] = None,
    ):
        """
        Initialize embeddings manager.

        Args:
            model_name: Model identifier (e.g., 'all-MiniLM-L6-v2')
            model_type: Type of embedding model
            device: Device for inference ('cpu', 'cuda', 'mps')
            cache_folder: Cache folder for models
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.cache_folder = cache_folder

        # Get full model name if using shorthand
        if model_name in self.RECOMMENDED_MODELS:
            self.full_model_name = self.RECOMMENDED_MODELS[model_name]["model_name"]
            self.dimensions = self.RECOMMENDED_MODELS[model_name]["dimensions"]
        else:
            self.full_model_name = model_name
            self.dimensions = None  # Will be determined at runtime

        self._embeddings = None

        LOGGER.info(
            f"Initialized EmbeddingsManager: {self.model_name} "
            f"(device={self.device})"
        )

    @property
    def embeddings(self):
        """Lazy-load embedding model."""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings

    def _create_embeddings(self):
        """Create embedding model instance."""
        if self.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
            LOGGER.info(f"Loading Sentence Transformers model: {self.full_model_name}")

            model_kwargs = {"device": self.device}
            if self.cache_folder:
                model_kwargs["cache_folder"] = self.cache_folder

            embeddings = HuggingFaceEmbeddings(
                model_name=self.full_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )

            LOGGER.info(f"✓ Model loaded successfully")
            return embeddings

        elif self.model_type == EmbeddingModelType.OPENAI:
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                raise ImportError("Install langchain-openai: pip install langchain-openai")

            LOGGER.info("Using OpenAI Embeddings (requires API key)")
            return OpenAIEmbeddings(model=self.model_name)

        elif self.model_type == EmbeddingModelType.HUGGINGFACE:
            LOGGER.info(f"Loading HuggingFace model: {self.full_model_name}")

            return HuggingFaceEmbeddings(
                model_name=self.full_model_name,
                model_kwargs={"device": self.device},
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)

    def get_model_info(self) -> dict:
        """Get model information."""
        info = {
            "model_name": self.model_name,
            "full_model_name": self.full_model_name,
            "model_type": self.model_type.value,
            "device": self.device,
        }

        if self.model_name in self.RECOMMENDED_MODELS:
            info.update(self.RECOMMENDED_MODELS[self.model_name])

        return info

    @classmethod
    def list_recommended_models(cls) -> dict:
        """List all recommended models with details."""
        return cls.RECOMMENDED_MODELS


# Singleton instance
_embeddings_manager_instance: Optional[EmbeddingsManager] = None


def get_embeddings_manager(
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs
) -> EmbeddingsManager:
    """
    Get or create embeddings manager singleton.

    Args:
        model_name: Model identifier
        **kwargs: Additional arguments for EmbeddingsManager

    Returns:
        EmbeddingsManager instance
    """
    global _embeddings_manager_instance

    if _embeddings_manager_instance is None:
        _embeddings_manager_instance = EmbeddingsManager(
            model_name=model_name,
            **kwargs
        )

    return _embeddings_manager_instance
