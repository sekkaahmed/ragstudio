"""
Re-ranking for Atlas-RAG.

Re-ranks retrieved documents using cross-encoder models for improved relevance.

Cross-encoders vs Bi-encoders:
- Bi-encoders (used in retrieval): Encode query and documents separately, fast but less accurate
- Cross-encoders (used in re-ranking): Encode query+document together, slower but more accurate

Workflow:
1. Retrieve top-K documents with fast bi-encoder (e.g., 20 documents)
2. Re-rank with accurate cross-encoder (e.g., return top-4)

Performance impact:
- Improves relevance by 10-30%
- Adds 50-200ms latency per query (depends on K)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


# Recommended cross-encoder models
RECOMMENDED_RERANKERS = {
    'ms-marco-MiniLM-L-6-v2': {
        'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'description': 'Fast, good for most use cases',
        'languages': ['en'],
        'size_mb': 80,
        'speed': 'fast',
        'quality': 'good',
    },
    'ms-marco-MiniLM-L-12-v2': {
        'model_name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'description': 'Better quality, slightly slower',
        'languages': ['en'],
        'size_mb': 130,
        'speed': 'medium',
        'quality': 'very good',
    },
    'mmarco-mMiniLMv2-L12-H384-v1': {
        'model_name': 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        'description': 'Multilingual (50+ languages including French)',
        'languages': ['multilingual'],
        'size_mb': 140,
        'speed': 'medium',
        'quality': 'very good',
    },
    'ms-marco-electra-base': {
        'model_name': 'cross-encoder/ms-marco-electra-base',
        'description': 'High quality, slower',
        'languages': ['en'],
        'size_mb': 420,
        'speed': 'slow',
        'quality': 'excellent',
    },
}


@dataclass
class RerankerConfig:
    """Configuration for re-ranking."""

    # Model name (from RECOMMENDED_RERANKERS or custom)
    model_name: str = 'ms-marco-MiniLM-L-6-v2'

    # Number of documents to return after re-ranking
    top_k: int = 4

    # Minimum relevance score (0.0 to 1.0)
    # Documents below this score are filtered out
    min_score: Optional[float] = None

    # Device for inference ('cpu', 'cuda', 'mps')
    device: str = 'cpu'

    # Batch size for re-ranking
    batch_size: int = 32


class Reranker:
    """
    Re-ranks documents using cross-encoder models.

    Cross-encoders provide more accurate relevance scores by jointly
    encoding the query and document, at the cost of higher latency.

    Example:
        >>> from src.core.rag.reranker import Reranker, RerankerConfig
        >>>
        >>> # Create reranker
        >>> config = RerankerConfig(
        ...     model_name='ms-marco-MiniLM-L-6-v2',
        ...     top_k=4,
        ...     min_score=0.5,
        ... )
        >>> reranker = Reranker(config)
        >>>
        >>> # Re-rank documents
        >>> query = "machine learning"
        >>> documents = [...]  # From initial retrieval
        >>> reranked_docs = reranker.rerank(query, documents)
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize reranker.

        Args:
            config: Reranker configuration
        """
        if not CROSSENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers not available. Install with: "
                "pip install sentence-transformers"
            )

        self.config = config or RerankerConfig()

        # Resolve model name
        if self.config.model_name in RECOMMENDED_RERANKERS:
            self.full_model_name = RECOMMENDED_RERANKERS[self.config.model_name]['model_name']
        else:
            self.full_model_name = self.config.model_name

        # Lazy initialization
        self._model = None

        LOGGER.info(
            f"Initialized Reranker: {self.config.model_name} "
            f"(device={self.config.device}, top_k={self.config.top_k})"
        )

    @property
    def model(self) -> CrossEncoder:
        """Get or load cross-encoder model (lazy loading)."""
        if self._model is None:
            LOGGER.info(f"Loading cross-encoder model: {self.full_model_name}")
            self._model = CrossEncoder(
                self.full_model_name,
                device=self.config.device,
            )
            LOGGER.info("✓ Model loaded successfully")

        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Re-rank documents by relevance to query.

        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of documents to return (overrides config)

        Returns:
            Re-ranked documents (top_k most relevant)
        """
        if not documents:
            return []

        top_k = top_k or self.config.top_k

        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Filter by min_score if set
        if self.config.min_score is not None:
            scored_docs = [
                (doc, score)
                for doc, score in scored_docs
                if score >= self.config.min_score
            ]

        # Get top-k
        top_docs = scored_docs[:top_k]

        # Add scores to metadata
        result_docs = []
        for doc, score in top_docs:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    'rerank_score': float(score),
                }
            )
            result_docs.append(doc_copy)

        LOGGER.debug(
            f"Re-ranked {len(documents)} documents → {len(result_docs)} "
            f"(scores: {[f'{s:.3f}' for _, s in top_docs]})"
        )

        return result_docs

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents and return with scores.

        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of documents to return

        Returns:
            List of (document, score) tuples
        """
        reranked_docs = self.rerank(query, documents, top_k)

        return [
            (doc, doc.metadata['rerank_score'])
            for doc in reranked_docs
        ]

    @staticmethod
    def list_recommended_models() -> dict:
        """List all recommended reranker models."""
        return RECOMMENDED_RERANKERS


def create_reranker(
    model_name: str = 'ms-marco-MiniLM-L-6-v2',
    top_k: int = 4,
    min_score: Optional[float] = None,
    device: str = 'cpu',
    **kwargs
) -> Reranker:
    """
    Helper function to create reranker.

    Args:
        model_name: Model name (from RECOMMENDED_RERANKERS or custom)
        top_k: Number of documents to return
        min_score: Minimum relevance score
        device: Device for inference
        **kwargs: Additional config parameters

    Returns:
        Configured Reranker

    Example:
        >>> reranker = create_reranker(
        ...     model_name='ms-marco-MiniLM-L-6-v2',
        ...     top_k=4,
        ...     min_score=0.5,
        ... )
    """
    config = RerankerConfig(
        model_name=model_name,
        top_k=top_k,
        min_score=min_score,
        device=device,
        **kwargs
    )

    return Reranker(config)


class RetrievalWithReranking:
    """
    Combines initial retrieval with re-ranking for optimal results.

    Two-stage approach:
    1. Retrieve top-N candidates with fast bi-encoder (e.g., N=20)
    2. Re-rank with accurate cross-encoder and return top-K (e.g., K=4)

    This gives quality of cross-encoder with speed closer to bi-encoder.

    Example:
        >>> from src.core.rag.vector_store import VectorStoreManager
        >>> from src.core.rag.reranker import RetrievalWithReranking
        >>>
        >>> pipeline = RetrievalWithReranking(
        ...     vector_store_manager=vsm,
        ...     reranker_model='ms-marco-MiniLM-L-6-v2',
        ...     initial_k=20,  # Retrieve 20 candidates
        ...     final_k=4,     # Return top 4 after re-ranking
        ... )
        >>>
        >>> results = pipeline.retrieve("machine learning")
    """

    def __init__(
        self,
        vector_store_manager,
        reranker_model: str = 'ms-marco-MiniLM-L-6-v2',
        initial_k: int = 20,
        final_k: int = 4,
        min_score: Optional[float] = None,
        device: str = 'cpu',
    ):
        """
        Initialize retrieval with re-ranking pipeline.

        Args:
            vector_store_manager: Vector store for initial retrieval
            reranker_model: Cross-encoder model for re-ranking
            initial_k: Number of candidates to retrieve initially
            final_k: Number of documents to return after re-ranking
            min_score: Minimum relevance score for re-ranking
            device: Device for re-ranker
        """
        self.vector_store_manager = vector_store_manager
        self.initial_k = initial_k
        self.final_k = final_k

        # Create reranker
        self.reranker = create_reranker(
            model_name=reranker_model,
            top_k=final_k,
            min_score=min_score,
            device=device,
        )

        LOGGER.info(
            f"Initialized RetrievalWithReranking: "
            f"initial_k={initial_k}, final_k={final_k}, "
            f"model={reranker_model}"
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve and re-rank documents.

        Args:
            query: Search query

        Returns:
            Top-K re-ranked documents
        """
        # Stage 1: Initial retrieval (fast)
        candidates = self.vector_store_manager.similarity_search(
            query,
            k=self.initial_k,
        )

        LOGGER.debug(f"Initial retrieval: {len(candidates)} candidates")

        if not candidates:
            return []

        # Stage 2: Re-ranking (accurate)
        reranked = self.reranker.rerank(query, candidates, top_k=self.final_k)

        LOGGER.debug(f"After re-ranking: {len(reranked)} documents")

        return reranked

    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve and re-rank documents with scores.

        Args:
            query: Search query

        Returns:
            List of (document, score) tuples
        """
        reranked = self.retrieve(query)
        return [
            (doc, doc.metadata.get('rerank_score', 0.0))
            for doc in reranked
        ]
