"""
Advanced retrievers for Atlas-RAG.

Provides various retrieval strategies:
- Basic similarity search
- Multi-query retrieval
- Contextual compression
- Ensemble retrieval
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import BaseRetriever, Document

from src.core.rag.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)


class AtlasRetriever:
    """
    Advanced retriever for Atlas-RAG.

    Provides multiple retrieval strategies with optional LLM enhancement.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        k: int = 4,
        score_threshold: float = 0.7,
    ):
        """
        Initialize retriever.

        Args:
            vector_store_manager: VectorStoreManager instance
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
        """
        self.vector_store_manager = vector_store_manager
        self.k = k
        self.score_threshold = score_threshold

        LOGGER.info(f"Initialized AtlasRetriever (k={k}, threshold={score_threshold})")

    def get_basic_retriever(self) -> BaseRetriever:
        """
        Get basic similarity search retriever.

        Returns:
            LangChain retriever
        """
        return self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={
                "k": self.k,
                "score_threshold": self.score_threshold,
            }
        )

    def get_multi_query_retriever(
        self,
        llm,
        num_queries: int = 3,
    ) -> MultiQueryRetriever:
        """
        Get multi-query retriever.

        Generates multiple queries from user input for better coverage.

        Args:
            llm: Language model for query generation
            num_queries: Number of queries to generate

        Returns:
            MultiQueryRetriever instance
        """
        LOGGER.info(f"Creating MultiQueryRetriever (num_queries={num_queries})")

        base_retriever = self.get_basic_retriever()

        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )

    def get_contextual_compression_retriever(
        self,
        llm=None,
    ) -> ContextualCompressionRetriever:
        """
        Get contextual compression retriever.

        Compresses retrieved documents to keep only relevant parts.

        Args:
            llm: Optional LLM for compression (if None, uses embeddings filter)

        Returns:
            ContextualCompressionRetriever instance
        """
        LOGGER.info("Creating ContextualCompressionRetriever")

        base_retriever = self.get_basic_retriever()

        # Use embeddings-based filter (no LLM needed)
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.vector_store_manager.embeddings_manager.embeddings,
            similarity_threshold=self.score_threshold,
        )

        redundant_filter = EmbeddingsRedundantFilter(
            embeddings=self.vector_store_manager.embeddings_manager.embeddings
        )

        pipeline = DocumentCompressorPipeline(
            transformers=[redundant_filter, embeddings_filter]
        )

        return ContextualCompressionRetriever(
            base_compressor=pipeline,
            base_retriever=base_retriever,
        )

    def retrieve(
        self,
        query: str,
        strategy: str = "basic",
        llm=None,
    ) -> List[Document]:
        """
        Retrieve documents using specified strategy.

        Args:
            query: Search query
            strategy: Retrieval strategy ('basic', 'multi_query', 'compression')
            llm: Optional LLM (required for some strategies)

        Returns:
            List of retrieved documents
        """
        LOGGER.info(f"Retrieving with strategy: {strategy}")

        if strategy == "basic":
            retriever = self.get_basic_retriever()
        elif strategy == "multi_query":
            if llm is None:
                raise ValueError("LLM required for multi_query strategy")
            retriever = self.get_multi_query_retriever(llm)
        elif strategy == "compression":
            retriever = self.get_contextual_compression_retriever(llm)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        results = retriever.get_relevant_documents(query)

        LOGGER.info(f"✓ Retrieved {len(results)} documents")
        return results


def create_retriever(
    vector_store_manager: VectorStoreManager,
    **kwargs
) -> AtlasRetriever:
    """
    Create Atlas retriever.

    Args:
        vector_store_manager: VectorStoreManager instance
        **kwargs: Additional arguments

    Returns:
        AtlasRetriever instance
    """
    return AtlasRetriever(
        vector_store_manager=vector_store_manager,
        **kwargs
    )
