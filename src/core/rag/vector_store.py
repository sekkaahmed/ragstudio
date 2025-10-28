"""
Vector store interface for Atlas-RAG.

Provides unified interface for different vector stores:
- Qdrant (recommended for production, local or cloud)
- JSON (simple file-based, good for dev/small datasets)
- FAISS (local, CPU-only)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Tuple, Literal
from pathlib import Path
import uuid

from langchain_community.vectorstores import Qdrant, FAISS

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    from langchain.schema import Document as LangChainDocument

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

from src.core.rag.embeddings import EmbeddingsManager
from src.core.storage.metadata_store import MetadataStore
from src.core.rag.json_vector_store import JSONVectorStore

LOGGER = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector store operations with metadata sync.

    Features:
    - Add documents with embeddings
    - Semantic search
    - Sync with metadata store (chunk.vector_id)
    - Support multiple vector stores
    """

    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        metadata_store: Optional[MetadataStore] = None,
        collection_name: str = "atlas_rag",
        backend_type: Literal["qdrant", "json"] = "qdrant",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        use_local: bool = True,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize vector store manager.

        Args:
            embeddings_manager: EmbeddingsManager instance
            metadata_store: MetadataStore for syncing vector_id
            collection_name: Name of vector collection
            backend_type: Type of backend ('qdrant' or 'json')
            qdrant_url: Qdrant server URL (if not local)
            qdrant_api_key: Qdrant API key (for cloud)
            use_local: Use local Qdrant instance (only for qdrant backend)
            persist_directory: Directory for JSON backend storage
        """
        self.embeddings_manager = embeddings_manager
        self.metadata_store = metadata_store
        self.collection_name = collection_name
        self.backend_type = backend_type
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.use_local = use_local
        self.persist_directory = persist_directory

        self._vector_store = None
        self._qdrant_client = None

        LOGGER.info(
            f"Initialized VectorStoreManager: {collection_name} "
            f"(backend={backend_type})"
        )

    @property
    def qdrant_client(self) -> QdrantClient:
        """Lazy-load Qdrant client."""
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant not installed: pip install qdrant-client")

        if self._qdrant_client is None:
            if self.use_local:
                # Local Qdrant (in-memory or persistent)
                LOGGER.info("Using local Qdrant (in-memory)")
                self._qdrant_client = QdrantClient(":memory:")
            else:
                # Remote Qdrant
                LOGGER.info(f"Connecting to Qdrant: {self.qdrant_url}")
                self._qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                )

            # Create collection if not exists
            self._ensure_collection_exists()

        return self._qdrant_client

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            LOGGER.info(f"Collection '{self.collection_name}' exists")
        except Exception:
            # Create collection
            model_info = self.embeddings_manager.get_model_info()
            dimensions = model_info.get("dimensions", 384)

            LOGGER.info(
                f"Creating collection '{self.collection_name}' "
                f"(dimensions={dimensions})"
            )

            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=Distance.COSINE,
                ),
            )

    @property
    def vector_store(self):
        """Lazy-load LangChain vector store."""
        if self._vector_store is None:
            if self.backend_type == "qdrant":
                self._vector_store = self._create_qdrant_store()
            elif self.backend_type == "json":
                self._vector_store = self._create_json_store()
            else:
                raise ValueError(f"Unknown backend_type: {self.backend_type}")

        return self._vector_store

    def _create_qdrant_store(self) -> Qdrant:
        """Create Qdrant vector store."""
        LOGGER.info("Creating LangChain Qdrant vector store")
        return Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embeddings_manager.embeddings,
        )

    def _create_json_store(self) -> JSONVectorStore:
        """Create JSON vector store."""
        LOGGER.info("Creating JSONVectorStore")

        persist_dir = self.persist_directory or f"./vector_data/{self.collection_name}"

        return JSONVectorStore(
            embedding=self.embeddings_manager.embeddings,
            persist_directory=persist_dir,
            collection_name=self.collection_name,
        )

    def add_documents(
        self,
        documents: List[LangChainDocument],
        chunk_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to vector store.

        Args:
            documents: List of LangChain documents
            chunk_ids: Optional list of chunk IDs from metadata store

        Returns:
            List of vector IDs
        """
        LOGGER.info(f"Adding {len(documents)} documents to vector store...")

        # Generate embeddings and add to vector store
        vector_ids = self.vector_store.add_documents(documents)

        LOGGER.info(f"✓ Added {len(vector_ids)} vectors to {self.collection_name}")

        # Sync with metadata store
        if self.metadata_store and chunk_ids:
            if len(chunk_ids) != len(vector_ids):
                LOGGER.warning(
                    f"Mismatch: {len(chunk_ids)} chunk_ids but {len(vector_ids)} vector_ids"
                )
            else:
                model_name = self.embeddings_manager.model_name

                for chunk_id, vector_id in zip(chunk_ids, vector_ids):
                    self.metadata_store.update_chunk_vector_id(
                        chunk_id=chunk_id,
                        vector_id=str(vector_id),
                        embedding_model=model_name,
                    )

                LOGGER.info("✓ Synced vector_ids with metadata store")

        return vector_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None,
    ) -> List[LangChainDocument]:
        """
        Semantic similarity search.

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of similar documents
        """
        LOGGER.debug(f"Searching for: '{query[:50]}...' (k={k})")

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict,
        )

        LOGGER.debug(f"✓ Found {len(results)} results")
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[LangChainDocument, float]]:
        """
        Semantic search with relevance scores.

        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of (document, score) tuples
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )

        # Filter by score threshold
        if score_threshold is not None:
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        LOGGER.debug(
            f"✓ Found {len(results)} results "
            f"(threshold={score_threshold})"
        )

        return results

    def delete_by_ids(self, vector_ids: List[str]):
        """Delete vectors by IDs."""
        LOGGER.info(f"Deleting {len(vector_ids)} vectors")
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=vector_ids,
        )

    def delete_collection(self):
        """Delete entire collection (use with caution!)."""
        LOGGER.warning(f"Deleting collection: {self.collection_name}")
        self.qdrant_client.delete_collection(self.collection_name)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""
        info = self.qdrant_client.get_collection(self.collection_name)

        return {
            "collection_name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }


def create_vector_store(
    embeddings_manager: EmbeddingsManager,
    metadata_store: Optional[MetadataStore] = None,
    **kwargs
) -> VectorStoreManager:
    """
    Create vector store manager.

    Args:
        embeddings_manager: EmbeddingsManager instance
        metadata_store: Optional MetadataStore for syncing
        **kwargs: Additional arguments

    Returns:
        VectorStoreManager instance
    """
    return VectorStoreManager(
        embeddings_manager=embeddings_manager,
        metadata_store=metadata_store,
        **kwargs
    )
