"""
Hybrid Search for Atlas-RAG.

Combines semantic similarity (vector search) with keyword matching (BM25)
for improved retrieval accuracy.

Benefits:
- Better recall: Captures both semantic meaning and exact keyword matches
- Robust to vocabulary mismatch: BM25 handles exact terms, vectors handle synonyms
- Configurable weighting: Balance between semantic and keyword search
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_community.retrievers import BM25Retriever
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from langchain.retrievers import EnsembleRetriever

from src.core.rag.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    # Weight for vector search (semantic similarity)
    # Range: 0.0 to 1.0
    # Higher = more weight on semantic similarity
    vector_weight: float = 0.5

    # Weight for BM25 search (keyword matching)
    # Range: 0.0 to 1.0
    # Higher = more weight on exact keyword matches
    bm25_weight: float = 0.5

    # Number of documents to retrieve
    k: int = 4

    # BM25 parameters
    # k1: Term frequency saturation parameter (typical: 1.2-2.0)
    bm25_k1: float = 1.5

    # b: Length normalization parameter (0 = no normalization, 1 = full normalization)
    bm25_b: float = 0.75

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.vector_weight <= 1.0:
            raise ValueError("vector_weight must be between 0.0 and 1.0")

        if not 0.0 <= self.bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be between 0.0 and 1.0")

        total_weight = self.vector_weight + self.bm25_weight
        if not 0.9 <= total_weight <= 1.1:
            LOGGER.warning(
                f"Weights don't sum to 1.0 (total={total_weight}). "
                "Results will be normalized."
            )


class HybridSearchRetriever:
    """
    Hybrid search combining vector similarity and BM25 keyword matching.

    Uses LangChain's EnsembleRetriever to merge results from:
    1. Vector store (semantic similarity via embeddings)
    2. BM25 (keyword matching with TF-IDF weighting)

    Example:
        >>> from src.core.rag.embeddings import get_embeddings_manager
        >>> from src.core.rag.vector_store import create_vector_store
        >>>
        >>> embeddings_manager = get_embeddings_manager()
        >>> vector_store_manager = create_vector_store(embeddings_manager)
        >>>
        >>> # Add documents
        >>> documents = [...]
        >>> vector_store_manager.add_documents(documents)
        >>>
        >>> # Create hybrid retriever
        >>> hybrid_retriever = HybridSearchRetriever(
        ...     vector_store_manager=vector_store_manager,
        ...     documents=documents,
        ...     config=HybridSearchConfig(vector_weight=0.6, bm25_weight=0.4)
        ... )
        >>>
        >>> # Search
        >>> results = hybrid_retriever.search("machine learning")
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        documents: List[Document],
        config: Optional[HybridSearchConfig] = None,
    ):
        """
        Initialize hybrid search retriever.

        Args:
            vector_store_manager: Vector store manager for semantic search
            documents: List of documents to index for BM25
            config: Hybrid search configuration
        """
        if not BM25_AVAILABLE:
            raise ImportError(
                "BM25Retriever not available. Install with: "
                "pip install langchain-community rank-bm25"
            )

        self.vector_store_manager = vector_store_manager
        self.documents = documents
        self.config = config or HybridSearchConfig()

        # Lazy initialization
        self._ensemble_retriever = None

        LOGGER.info(
            f"Initialized HybridSearchRetriever: "
            f"vector_weight={self.config.vector_weight}, "
            f"bm25_weight={self.config.bm25_weight}, "
            f"documents={len(documents)}"
        )

    @property
    def ensemble_retriever(self) -> EnsembleRetriever:
        """Get or create ensemble retriever (lazy loading)."""
        if self._ensemble_retriever is None:
            self._ensemble_retriever = self._create_ensemble_retriever()
        return self._ensemble_retriever

    def _create_ensemble_retriever(self) -> EnsembleRetriever:
        """Create ensemble retriever combining vector and BM25."""
        # 1. Vector retriever
        vector_retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={'k': self.config.k}
        )

        # 2. BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents=self.documents,
            k=self.config.k,
        )

        # Set BM25 parameters if available
        if hasattr(bm25_retriever, 'vectorizer'):
            # Note: rank-bm25 library may not expose k1/b directly
            # This is a best-effort attempt
            try:
                bm25_retriever.k1 = self.config.bm25_k1
                bm25_retriever.b = self.config.bm25_b
            except AttributeError:
                LOGGER.debug("BM25 parameters k1/b not settable for this implementation")

        # 3. Ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[self.config.vector_weight, self.config.bm25_weight],
        )

        LOGGER.info(
            f"Created ensemble retriever: "
            f"vector={self.config.vector_weight}, bm25={self.config.bm25_weight}"
        )

        return ensemble_retriever

    def search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """
        Search using hybrid approach.

        Args:
            query: Search query
            k: Number of documents to retrieve (overrides config)

        Returns:
            List of documents ranked by hybrid score
        """
        # Update k if provided
        if k is not None:
            # Need to recreate retriever with new k
            self.config.k = k
            self._ensemble_retriever = None

        results = self.ensemble_retriever.invoke(query)

        LOGGER.debug(
            f"Hybrid search for '{query[:50]}...' returned {len(results)} documents"
        )

        return results

    def update_documents(self, documents: List[Document]):
        """
        Update the document index for BM25.

        Call this when documents are added/removed from vector store.

        Args:
            documents: Updated list of all documents
        """
        self.documents = documents
        self._ensemble_retriever = None  # Force recreation

        LOGGER.info(f"Updated document index: {len(documents)} documents")

    def get_info(self) -> Dict[str, Any]:
        """Get retriever information."""
        return {
            'type': 'hybrid_search',
            'vector_weight': self.config.vector_weight,
            'bm25_weight': self.config.bm25_weight,
            'k': self.config.k,
            'num_documents': len(self.documents),
            'bm25_k1': self.config.bm25_k1,
            'bm25_b': self.config.bm25_b,
        }


def create_hybrid_retriever(
    vector_store_manager: VectorStoreManager,
    documents: List[Document],
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    k: int = 4,
    **kwargs
) -> HybridSearchRetriever:
    """
    Helper function to create hybrid search retriever.

    Args:
        vector_store_manager: Vector store manager
        documents: Documents to index
        vector_weight: Weight for vector search (0.0-1.0)
        bm25_weight: Weight for BM25 search (0.0-1.0)
        k: Number of documents to retrieve
        **kwargs: Additional config parameters

    Returns:
        Configured HybridSearchRetriever

    Example:
        >>> retriever = create_hybrid_retriever(
        ...     vector_store_manager=vsm,
        ...     documents=docs,
        ...     vector_weight=0.7,
        ...     bm25_weight=0.3,
        ... )
    """
    config = HybridSearchConfig(
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        k=k,
        **kwargs
    )

    return HybridSearchRetriever(
        vector_store_manager=vector_store_manager,
        documents=documents,
        config=config,
    )


# Preset configurations for different use cases
HYBRID_PRESETS = {
    'balanced': HybridSearchConfig(
        vector_weight=0.5,
        bm25_weight=0.5,
    ),
    'semantic_focused': HybridSearchConfig(
        vector_weight=0.7,
        bm25_weight=0.3,
    ),
    'keyword_focused': HybridSearchConfig(
        vector_weight=0.3,
        bm25_weight=0.7,
    ),
    'semantic_heavy': HybridSearchConfig(
        vector_weight=0.8,
        bm25_weight=0.2,
    ),
    'keyword_heavy': HybridSearchConfig(
        vector_weight=0.2,
        bm25_weight=0.8,
    ),
}


def get_hybrid_preset(preset_name: str) -> HybridSearchConfig:
    """
    Get a preset configuration for hybrid search.

    Available presets:
    - 'balanced': Equal weight (0.5/0.5)
    - 'semantic_focused': More semantic (0.7/0.3)
    - 'keyword_focused': More keywords (0.3/0.7)
    - 'semantic_heavy': Heavy semantic (0.8/0.2)
    - 'keyword_heavy': Heavy keywords (0.2/0.8)

    Args:
        preset_name: Name of the preset

    Returns:
        HybridSearchConfig

    Raises:
        ValueError: If preset not found

    Example:
        >>> config = get_hybrid_preset('semantic_focused')
        >>> retriever = HybridSearchRetriever(vsm, docs, config)
    """
    if preset_name not in HYBRID_PRESETS:
        available = ', '.join(HYBRID_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {available}"
        )

    return HYBRID_PRESETS[preset_name]
