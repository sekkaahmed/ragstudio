"""
Pinecone vector database connector.

Pinecone is a fully-managed vector database optimized for similarity search.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.workflows.io.schema import Chunk
from src.core.vector.base import VectorStore, VectorStoreConfig

LOGGER = logging.getLogger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None
    ServerlessSpec = None


class PineconeStore(VectorStore):
    """Pinecone vector database connector."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize Pinecone store.

        Args:
            config: Vector store configuration
                - api_key: Pinecone API key (required)
                - index_name: Pinecone index name
                - extra_config: Additional Pinecone settings
                    - environment: Pinecone environment (e.g., "us-west1-gcp")
                    - cloud: Cloud provider (e.g., "aws", "gcp", "azure")
                    - region: Cloud region
        """
        super().__init__(config)

        if not PINECONE_AVAILABLE:
            raise ImportError(
                "pinecone-client is not installed. "
                "Install with: pip install pinecone-client"
            )

        if not config.api_key:
            raise ValueError("Pinecone API key is required")

        self.index = None

    def connect(self) -> None:
        """Establish connection to Pinecone."""
        # Initialize Pinecone client
        self.client = Pinecone(api_key=self.config.api_key)

        LOGGER.info("Connected to Pinecone")

        # Connect to index if it exists
        if self.index_exists():
            self.index = self.client.Index(self.config.index_name)
            LOGGER.info("Connected to Pinecone index '%s'", self.config.index_name)

    def create_index(
        self,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        **kwargs
    ) -> None:
        """
        Create a new index in Pinecone.

        Args:
            index_name: Index name (defaults to config.index_name)
            dimension: Vector dimension (defaults to config.embedding_dimension)
            metric: Distance metric ("cosine", "euclidean", "dotproduct")
            cloud: Cloud provider ("aws", "gcp", "azure")
            region: Cloud region
            **kwargs: Additional Pinecone-specific parameters
        """
        if self.client is None:
            self.connect()

        index_name = index_name or self.config.index_name
        vector_dimension = dimension or self.config.embedding_dimension

        # Get cloud and region from extra_config if available
        if self.config.extra_config:
            cloud = self.config.extra_config.get("cloud", cloud)
            region = self.config.extra_config.get("region", region)

        # Create serverless index
        self.client.create_index(
            name=index_name,
            dimension=vector_dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region,
            ),
        )

        # Connect to the newly created index
        self.index = self.client.Index(index_name)

        LOGGER.info(
            "Created Pinecone index '%s' with dimension %d and %s metric",
            index_name,
            vector_dimension,
            metric,
        )

    def index_exists(self, index_name: Optional[str] = None) -> bool:
        """Check if an index exists."""
        if self.client is None:
            self.connect()

        index_name = index_name or self.config.index_name

        try:
            index_list = self.client.list_indexes()
            # Check if index_name is in the list
            return any(idx.name == index_name for idx in index_list)
        except Exception as e:
            LOGGER.warning("Error checking if index exists: %s", e)
            return False

    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        namespace: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert or update chunks in Pinecone.

        Args:
            chunks: List of chunks to upsert
            embeddings: List of embedding vectors
            namespace: Pinecone namespace (optional, for partitioning)
            **kwargs: Additional Pinecone-specific parameters

        Returns:
            Dictionary with upsert results
        """
        if self.client is None:
            self.connect()

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # Ensure index exists
        if not self.index_exists():
            self.create_index()

        if self.index is None:
            self.index = self.client.Index(self.config.index_name)

        # Convert chunks to Pinecone vectors
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            # Prepare metadata (Pinecone has metadata size limits)
            metadata = {
                "text": chunk.text[:40000],  # Pinecone metadata limit
                "document_id": chunk.document_id,
            }

            # Add specific metadata fields if configured
            if self.config.metadata_fields:
                for field in self.config.metadata_fields:
                    if field in chunk.metadata:
                        value = chunk.metadata[field]
                        # Ensure value is JSON-serializable
                        if isinstance(value, (str, int, float, bool)):
                            metadata[field] = value
                        elif isinstance(value, (list, dict)):
                            metadata[field] = str(value)

            vectors.append({
                "id": chunk.id,
                "values": embedding,
                "metadata": metadata,
            })

        # Upsert in batches
        batch_size = self.config.batch_size
        total_upserted = 0

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(
                vectors=batch,
                namespace=namespace,
            )
            total_upserted += len(batch)

        LOGGER.info(
            "Upserted %d chunks to Pinecone index '%s'",
            total_upserted,
            self.config.index_name,
        )

        return {
            "count": total_upserted,
            "index": self.config.index_name,
            "namespace": namespace,
        }

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        include_metadata: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Pinecone.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"document_id": "doc123"})
            namespace: Pinecone namespace to search in
            include_metadata: Whether to include metadata in results
            **kwargs: Additional Pinecone-specific search parameters

        Returns:
            List of search results with scores and metadata
        """
        if self.index is None:
            if self.client is None:
                self.connect()
            self.index = self.client.Index(self.config.index_name)

        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            namespace=namespace,
            include_metadata=include_metadata,
            **kwargs
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            result = {
                "id": match.id,
                "score": match.score,
            }

            if include_metadata and hasattr(match, 'metadata'):
                result["text"] = match.metadata.get("text", "")
                result["document_id"] = match.metadata.get("document_id", "")
                result["metadata"] = match.metadata

            formatted_results.append(result)

        LOGGER.debug("Found %d results for query", len(formatted_results))

        return formatted_results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        delete_all: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delete vectors from Pinecone.

        Args:
            ids: List of chunk IDs to delete
            filter_dict: Metadata filters for deletion
            namespace: Pinecone namespace
            delete_all: Delete all vectors in namespace (dangerous!)
            **kwargs: Additional Pinecone-specific delete parameters

        Returns:
            Dictionary with deletion results
        """
        if self.index is None:
            if self.client is None:
                self.connect()
            self.index = self.client.Index(self.config.index_name)

        if delete_all:
            self.index.delete(delete_all=True, namespace=namespace)
            count = "all"
        elif ids:
            self.index.delete(ids=ids, namespace=namespace)
            count = len(ids)
        elif filter_dict:
            self.index.delete(filter=filter_dict, namespace=namespace)
            count = "unknown"  # Pinecone doesn't return delete count with filters
        else:
            raise ValueError("Either ids, filter_dict, or delete_all must be provided")

        LOGGER.info("Deleted %s vectors from Pinecone", count)

        return {
            "count": count,
            "index": self.config.index_name,
            "namespace": namespace,
        }

    def close(self) -> None:
        """Close connection to Pinecone."""
        # Pinecone client doesn't require explicit close
        self.client = None
        self.index = None
        LOGGER.info("Closed Pinecone connection")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        stats = super().get_stats()

        if self.index is not None:
            try:
                index_stats = self.index.describe_index_stats()
                stats.update({
                    "total_vector_count": index_stats.total_vector_count,
                    "dimension": index_stats.dimension,
                    "namespaces": index_stats.namespaces,
                })
            except Exception as e:
                LOGGER.warning("Error getting Pinecone stats: %s", e)

        return stats
