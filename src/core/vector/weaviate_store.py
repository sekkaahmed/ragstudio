"""
Weaviate vector database connector.

Weaviate is an open-source vector database with GraphQL API and ML model integration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.workflows.io.schema import Chunk
from src.core.vector.base import VectorStore, VectorStoreConfig

LOGGER = logging.getLogger(__name__)

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None


class WeaviateStore(VectorStore):
    """Weaviate vector database connector."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize Weaviate store.

        Args:
            config: Vector store configuration
                - url: Weaviate server URL (default: "http://localhost:8080")
                - api_key: Weaviate API key (optional)
                - index_name: Collection/Class name
                - extra_config: Additional Weaviate settings
                    - grpc_port: gRPC port (default: 50051)
        """
        super().__init__(config)

        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "weaviate-client is not installed. "
                "Install with: pip install weaviate-client"
            )

        self.class_name = self._format_class_name(config.index_name)

    @staticmethod
    def _format_class_name(name: str) -> str:
        """Format class name to meet Weaviate requirements (PascalCase)."""
        # Remove special characters and split on non-alphanumeric
        import re
        words = re.findall(r'[a-zA-Z0-9]+', name)
        # Capitalize first letter of each word
        return ''.join(word.capitalize() for word in words)

    def connect(self) -> None:
        """Establish connection to Weaviate."""
        url = self.config.url or "http://localhost:8080"

        if self.config.api_key:
            # Connect with API key
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(self.config.api_key),
            )
        else:
            # Connect to local Weaviate
            self.client = weaviate.connect_to_local(
                host=url.replace("http://", "").replace("https://", "").split(":")[0],
                port=int(url.split(":")[-1]) if ":" in url else 8080,
            )

        LOGGER.info("Connected to Weaviate at %s", url)

    def create_index(
        self,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        distance: str = "cosine",
        **kwargs
    ) -> None:
        """
        Create a new collection in Weaviate.

        Args:
            index_name: Collection name (defaults to config.index_name)
            dimension: Vector dimension (defaults to config.embedding_dimension)
            distance: Distance metric ("cosine", "dot", "l2-squared", "hamming", "manhattan")
            **kwargs: Additional Weaviate-specific parameters
        """
        if self.client is None:
            self.connect()

        class_name = self._format_class_name(index_name or self.config.index_name)
        vector_dimension = dimension or self.config.embedding_dimension

        # Create collection with properties
        self.client.collections.create(
            name=class_name,
            vectorizer_config=Configure.Vectorizer.none(),  # We provide embeddings
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=distance,
            ),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="metadata_json", data_type=DataType.TEXT),
            ],
        )

        LOGGER.info(
            "Created Weaviate collection '%s' with dimension %d and %s distance",
            class_name,
            vector_dimension,
            distance,
        )

    def index_exists(self, index_name: Optional[str] = None) -> bool:
        """Check if a collection exists."""
        if self.client is None:
            self.connect()

        class_name = self._format_class_name(index_name or self.config.index_name)

        return self.client.collections.exists(class_name)

    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Insert or update chunks in Weaviate.

        Args:
            chunks: List of chunks to upsert
            embeddings: List of embedding vectors
            **kwargs: Additional Weaviate-specific parameters

        Returns:
            Dictionary with upsert results
        """
        if self.client is None:
            self.connect()

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # Ensure collection exists
        if not self.index_exists():
            self.create_index()

        collection = self.client.collections.get(self.class_name)

        # Convert chunks to Weaviate objects
        import json
        objects_to_insert = []

        for chunk, embedding in zip(chunks, embeddings):
            # Prepare properties
            properties = {
                "text": chunk.text,
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "metadata_json": json.dumps(chunk.metadata),
            }

            # Add specific metadata fields if configured
            if self.config.metadata_fields:
                for field in self.config.metadata_fields:
                    if field in chunk.metadata:
                        value = chunk.metadata[field]
                        # Weaviate supports specific types
                        if isinstance(value, (str, int, float, bool)):
                            properties[field] = value

            objects_to_insert.append({
                "properties": properties,
                "vector": embedding,
                "uuid": chunk.id,  # Use chunk ID as UUID
            })

        # Batch insert
        with collection.batch.dynamic() as batch:
            for obj in objects_to_insert:
                batch.add_object(
                    properties=obj["properties"],
                    vector=obj["vector"],
                    uuid=obj["uuid"],
                )

        total_upserted = len(objects_to_insert)

        LOGGER.info(
            "Upserted %d chunks to Weaviate collection '%s'",
            total_upserted,
            self.class_name,
        )

        return {
            "count": total_upserted,
            "collection": self.class_name,
        }

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Weaviate.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"document_id": "doc123"})
            **kwargs: Additional Weaviate-specific search parameters

        Returns:
            List of search results with scores and metadata
        """
        if self.client is None:
            self.connect()

        collection = self.client.collections.get(self.class_name)

        # Build Weaviate filter
        query_filter = None
        if filter_dict:
            # Simple equality filters
            from weaviate.classes.query import Filter as WeaviateFilter
            filters = []
            for key, value in filter_dict.items():
                filters.append(
                    WeaviateFilter.by_property(key).equal(value)
                )
            # Combine filters with AND
            if filters:
                query_filter = filters[0]
                for f in filters[1:]:
                    query_filter = query_filter & f

        # Search
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            filters=query_filter,
            return_metadata=["distance"],
            **kwargs
        )

        # Format results
        import json
        formatted_results = []

        for obj in response.objects:
            # Calculate similarity score from distance (assuming cosine)
            distance = obj.metadata.distance
            score = 1.0 - distance  # Convert distance to similarity

            result = {
                "id": str(obj.uuid),
                "score": score,
                "text": obj.properties.get("text", ""),
                "document_id": obj.properties.get("document_id", ""),
            }

            # Parse metadata JSON
            metadata_json = obj.properties.get("metadata_json", "{}")
            try:
                result["metadata"] = json.loads(metadata_json)
            except json.JSONDecodeError:
                result["metadata"] = {}

            formatted_results.append(result)

        LOGGER.debug("Found %d results for query", len(formatted_results))

        return formatted_results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Delete vectors from Weaviate.

        Args:
            ids: List of chunk IDs to delete
            filter_dict: Metadata filters for deletion
            **kwargs: Additional Weaviate-specific delete parameters

        Returns:
            Dictionary with deletion results
        """
        if self.client is None:
            self.connect()

        collection = self.client.collections.get(self.class_name)

        if ids:
            # Delete by IDs
            for chunk_id in ids:
                collection.data.delete_by_id(chunk_id)
            count = len(ids)
        elif filter_dict:
            # Delete by filter
            from weaviate.classes.query import Filter as WeaviateFilter
            filters = []
            for key, value in filter_dict.items():
                filters.append(
                    WeaviateFilter.by_property(key).equal(value)
                )
            query_filter = filters[0]
            for f in filters[1:]:
                query_filter = query_filter & f

            result = collection.data.delete_many(where=query_filter)
            count = result.successful if hasattr(result, 'successful') else "unknown"
        else:
            raise ValueError("Either ids or filter_dict must be provided")

        LOGGER.info("Deleted %s vectors from Weaviate", count)

        return {
            "count": count,
            "collection": self.class_name,
        }

    def close(self) -> None:
        """Close connection to Weaviate."""
        if self.client is not None:
            self.client.close()
            self.client = None
            LOGGER.info("Closed Weaviate connection")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Weaviate collection."""
        stats = super().get_stats()

        if self.client and self.index_exists():
            try:
                collection = self.client.collections.get(self.class_name)
                aggregate_result = collection.aggregate.over_all()

                stats.update({
                    "total_count": aggregate_result.total_count,
                })
            except Exception as e:
                LOGGER.warning("Error getting Weaviate stats: %s", e)

        return stats
