"""
Enhanced Vector Database Connectors for Production RAG
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import json
from datetime import datetime

# Qdrant imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Weaviate imports
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

# Pinecone imports
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorStoreBase(ABC):
    """Base class for vector database operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy"""
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str, vector_size: int = 384) -> bool:
        """Create a new collection"""
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections"""
        pass
    
    @abstractmethod
    async def store_chunk(
        self, 
        chunk_id: str, 
        text: str, 
        embedding: List[float], 
        metadata: Dict[str, Any],
        collection_name: str
    ) -> bool:
        """Store a chunk with its embedding"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        collection_name: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    async def batch_store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str
    ) -> int:
        """Store multiple chunks in batch"""
        pass

class QdrantStore(VectorStoreBase):
    """Enhanced Qdrant vector store connector"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        config = config or {
            "host": "localhost",
            "port": 6333,
            "timeout": 60
        }
        
        super().__init__(config)
        
        self.client = QdrantClient(
            host=config["host"],
            port=config["port"],
            timeout=config.get("timeout", 60)
        )
        
        self.logger.info(f"Qdrant client initialized: {config['host']}:{config['port']}")
    
    async def health_check(self) -> bool:
        """Check Qdrant health"""
        try:
            collections = self.client.get_collections()
            self.logger.debug("Qdrant health check passed")
            return True
        except Exception as e:
            self.logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def create_collection(self, collection_name: str, vector_size: int = 384) -> bool:
        """Create a new Qdrant collection"""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload index for metadata filtering
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.logger.info(f"Created Qdrant collection: {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create Qdrant collection {collection_name}: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a Qdrant collection"""
        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f"Deleted Qdrant collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete Qdrant collection {collection_name}: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List all Qdrant collections"""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            self.logger.error(f"Failed to list Qdrant collections: {e}")
            return []
    
    async def store_chunk(
        self, 
        chunk_id: str, 
        text: str, 
        embedding: List[float], 
        metadata: Dict[str, Any],
        collection_name: str
    ) -> bool:
        """Store a chunk in Qdrant"""
        try:
            # Prepare payload
            payload = {
                "text": text,
                "chunk_id": chunk_id,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }
            
            # Create point
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=payload
            )
            
            # Store in collection
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            self.logger.debug(f"Stored chunk {chunk_id} in Qdrant collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store chunk {chunk_id} in Qdrant: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        collection_name: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in Qdrant"""
        try:
            # Prepare filter
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            # Perform search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                    "chunk_id": result.payload.get("chunk_id", result.id)
                })
            
            self.logger.debug(f"Found {len(results)} results in Qdrant collection {collection_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Qdrant search failed: {e}")
            return []
    
    async def batch_store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str
    ) -> int:
        """Store multiple chunks in batch"""
        try:
            points = []
            for chunk in chunks:
                point = PointStruct(
                    id=chunk["id"],
                    vector=chunk["embedding"],
                    payload={
                        "text": chunk["text"],
                        "chunk_id": chunk["id"],
                        "timestamp": datetime.now().isoformat(),
                        **chunk.get("metadata", {})
                    }
                )
                points.append(point)
            
            # Batch upsert
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            self.logger.info(f"Batch stored {len(points)} chunks in Qdrant collection {collection_name}")
            return len(points)
            
        except Exception as e:
            self.logger.error(f"Batch store failed in Qdrant: {e}")
            return 0

class WeaviateStore(VectorStoreBase):
    """Enhanced Weaviate vector store connector"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate client not available. Install with: pip install weaviate-client")
        
        config = config or {
            "url": "http://localhost:8080"
        }
        
        super().__init__(config)
        
        self.client = weaviate.Client(config["url"])
        self.logger.info(f"Weaviate client initialized: {config['url']}")
    
    async def health_check(self) -> bool:
        """Check Weaviate health"""
        try:
            self.client.is_ready()
            self.logger.debug("Weaviate health check passed")
            return True
        except Exception as e:
            self.logger.error(f"Weaviate health check failed: {e}")
            return False
    
    async def create_collection(self, collection_name: str, vector_size: int = 384) -> bool:
        """Create a new Weaviate collection"""
        try:
            class_schema = {
                "class": collection_name,
                "description": f"ChunkForge collection: {collection_name}",
                "vectorizer": "none",  # We provide our own embeddings
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Chunk text content"
                    },
                    {
                        "name": "chunk_id",
                        "dataType": ["string"],
                        "description": "Unique chunk identifier"
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                        "description": "Source document"
                    },
                    {
                        "name": "document_id",
                        "dataType": ["string"],
                        "description": "Document identifier"
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["date"],
                        "description": "Processing timestamp"
                    }
                ]
            }
            
            self.client.schema.create_class(class_schema)
            self.logger.info(f"Created Weaviate collection: {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create Weaviate collection {collection_name}: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a Weaviate collection"""
        try:
            self.client.schema.delete_class(collection_name)
            self.logger.info(f"Deleted Weaviate collection: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete Weaviate collection {collection_name}: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List all Weaviate collections"""
        try:
            schema = self.client.schema.get()
            return [cls["class"] for cls in schema["classes"]]
        except Exception as e:
            self.logger.error(f"Failed to list Weaviate collections: {e}")
            return []
    
    async def store_chunk(
        self, 
        chunk_id: str, 
        text: str, 
        embedding: List[float], 
        metadata: Dict[str, Any],
        collection_name: str
    ) -> bool:
        """Store a chunk in Weaviate"""
        try:
            data_object = {
                "text": text,
                "chunk_id": chunk_id,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }
            
            self.client.data_object.create(
                data_object=data_object,
                class_name=collection_name,
                uuid=chunk_id,
                vector=embedding
            )
            
            self.logger.debug(f"Stored chunk {chunk_id} in Weaviate collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store chunk {chunk_id} in Weaviate: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        collection_name: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in Weaviate"""
        try:
            # Prepare where clause
            where_clause = None
            if filter_conditions:
                where_clause = {
                    "operator": "And",
                    "operands": [
                        {"path": [key], "operator": "Equal", "valueString": value}
                        for key, value in filter_conditions.items()
                    ]
                }
            
            # Perform search
            result = self.client.query.get(
                class_name=collection_name,
                properties=["text", "chunk_id", "source", "document_id", "timestamp"]
            ).with_near_vector({
                "vector": query_embedding,
                "certainty": score_threshold
            }).with_limit(limit)
            
            if where_clause:
                result = result.with_where(where_clause)
            
            search_results = result.do()
            
            # Format results
            results = []
            if search_results and "data" in search_results and "Get" in search_results["data"]:
                for item in search_results["data"]["Get"][collection_name]:
                    results.append({
                        "id": item.get("_additional", {}).get("id", ""),
                        "score": item.get("_additional", {}).get("certainty", 0),
                        "text": item.get("text", ""),
                        "metadata": {
                            "chunk_id": item.get("chunk_id", ""),
                            "source": item.get("source", ""),
                            "document_id": item.get("document_id", ""),
                            "timestamp": item.get("timestamp", "")
                        }
                    })
            
            self.logger.debug(f"Found {len(results)} results in Weaviate collection {collection_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Weaviate search failed: {e}")
            return []
    
    async def batch_store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str
    ) -> int:
        """Store multiple chunks in batch"""
        try:
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for chunk in chunks:
                    data_object = {
                        "text": chunk["text"],
                        "chunk_id": chunk["id"],
                        "timestamp": datetime.now().isoformat(),
                        **chunk.get("metadata", {})
                    }
                    
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=collection_name,
                        uuid=chunk["id"],
                        vector=chunk["embedding"]
                    )
            
            self.logger.info(f"Batch stored {len(chunks)} chunks in Weaviate collection {collection_name}")
            return len(chunks)
            
        except Exception as e:
            self.logger.error(f"Batch store failed in Weaviate: {e}")
            return 0

class PineconeStore(VectorStoreBase):
    """Enhanced Pinecone vector store connector"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available. Install with: pip install pinecone-client")
        
        config = config or {
            "api_key": None,
            "environment": "us-west1-gcp"
        }
        
        super().__init__(config)
        
        if not config.get("api_key"):
            raise ValueError("Pinecone API key is required")
        
        self.pc = Pinecone(api_key=config["api_key"])
        self.logger.info("Pinecone client initialized")
    
    async def health_check(self) -> bool:
        """Check Pinecone health"""
        try:
            self.pc.list_indexes()
            self.logger.debug("Pinecone health check passed")
            return True
        except Exception as e:
            self.logger.error(f"Pinecone health check failed: {e}")
            return False
    
    async def create_collection(self, collection_name: str, vector_size: int = 384) -> bool:
        """Create a new Pinecone index"""
        try:
            self.pc.create_index(
                name=collection_name,
                dimension=vector_size,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            self.logger.info(f"Created Pinecone index: {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create Pinecone index {collection_name}: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a Pinecone index"""
        try:
            self.pc.delete_index(collection_name)
            self.logger.info(f"Deleted Pinecone index: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete Pinecone index {collection_name}: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List all Pinecone indexes"""
        try:
            indexes = self.pc.list_indexes()
            return [index.name for index in indexes]
        except Exception as e:
            self.logger.error(f"Failed to list Pinecone indexes: {e}")
            return []
    
    async def store_chunk(
        self, 
        chunk_id: str, 
        text: str, 
        embedding: List[float], 
        metadata: Dict[str, Any],
        collection_name: str
    ) -> bool:
        """Store a chunk in Pinecone"""
        try:
            index = self.pc.Index(collection_name)
            
            # Prepare metadata
            pinecone_metadata = {
                "text": text,
                "chunk_id": chunk_id,
                "timestamp": datetime.now().isoformat(),
                **metadata
            }
            
            # Store vector
            index.upsert(
                vectors=[(chunk_id, embedding, pinecone_metadata)]
            )
            
            self.logger.debug(f"Stored chunk {chunk_id} in Pinecone index {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store chunk {chunk_id} in Pinecone: {e}")
            return False
    
    async def search(
        self,
        query_embedding: List[float],
        collection_name: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in Pinecone"""
        try:
            index = self.pc.Index(collection_name)
            
            # Perform search
            search_results = index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True,
                filter=filter_conditions
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                if match.score >= score_threshold:
                    results.append({
                        "id": match.id,
                        "score": match.score,
                        "text": match.metadata.get("text", ""),
                        "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
                        "chunk_id": match.metadata.get("chunk_id", match.id)
                    })
            
            self.logger.debug(f"Found {len(results)} results in Pinecone index {collection_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Pinecone search failed: {e}")
            return []
    
    async def batch_store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str
    ) -> int:
        """Store multiple chunks in batch"""
        try:
            index = self.pc.Index(collection_name)
            
            # Prepare vectors
            vectors = []
            for chunk in chunks:
                metadata = {
                    "text": chunk["text"],
                    "chunk_id": chunk["id"],
                    "timestamp": datetime.now().isoformat(),
                    **chunk.get("metadata", {})
                }
                
                vectors.append((chunk["id"], chunk["embedding"], metadata))
            
            # Batch upsert
            index.upsert(vectors=vectors)
            
            self.logger.info(f"Batch stored {len(vectors)} chunks in Pinecone index {collection_name}")
            return len(vectors)
            
        except Exception as e:
            self.logger.error(f"Batch store failed in Pinecone: {e}")
            return 0

# Factory function
def create_vector_store(store_type: str, config: Optional[Dict[str, Any]] = None) -> VectorStoreBase:
    """Factory function to create vector store instances"""
    if store_type.lower() == "qdrant":
        return QdrantStore(config)
    elif store_type.lower() == "weaviate":
        return WeaviateStore(config)
    elif store_type.lower() == "pinecone":
        return PineconeStore(config)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
