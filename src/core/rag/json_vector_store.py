"""
JSON-based Vector Store for Atlas-RAG.

Provides a simple file-based vector store as an alternative to Qdrant.
Useful for:
- Development and testing (no infrastructure needed)
- Small datasets (< 10k documents)
- Export/backup of vector data
- Sharing datasets
- Debugging and inspection

Storage format:
- documents.json: List of documents with metadata
- embeddings.npy: NumPy array of embeddings
- index.json: Metadata about the store

Limitations:
- Not optimized for large scale (use Qdrant for production)
- No distributed search
- Slower than dedicated vector DB
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
except ImportError:
    from langchain.schema import Document
    from langchain.vectorstores.base import VectorStore

LOGGER = logging.getLogger(__name__)


class JSONVectorStore(VectorStore):
    """
    Simple JSON-based vector store.

    Stores documents and embeddings in JSON/NumPy files.
    Implements basic similarity search using cosine similarity.

    Example:
        >>> from src.core.rag.embeddings import get_embeddings_manager
        >>> from src.core.rag.json_vector_store import JSONVectorStore
        >>>
        >>> embeddings = get_embeddings_manager()
        >>> store = JSONVectorStore(
        ...     embedding=embeddings,
        ...     persist_directory="./vector_data"
        ... )
        >>>
        >>> # Add documents
        >>> store.add_documents([
        ...     Document(page_content="Text 1", metadata={}),
        ...     Document(page_content="Text 2", metadata={}),
        ... ])
        >>>
        >>> # Search
        >>> results = store.similarity_search("query", k=2)
        >>>
        >>> # Save
        >>> store.persist()
        >>>
        >>> # Load later
        >>> store2 = JSONVectorStore.load(
        ...     persist_directory="./vector_data",
        ...     embedding=embeddings
        ... )
    """

    def __init__(
        self,
        embedding,
        persist_directory: Optional[str] = None,
        collection_name: str = "default",
    ):
        """
        Initialize JSON vector store.

        Args:
            embedding: Embeddings instance (must have embed_documents/embed_query methods)
            persist_directory: Directory to save files
            collection_name: Name of the collection
        """
        self._embedding = embedding
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.collection_name = collection_name

        # In-memory storage
        self._documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None
        self._document_ids: List[str] = []

        # Try to load existing data
        if self.persist_directory and self.persist_directory.exists():
            try:
                self._load_from_disk()
            except Exception as e:
                LOGGER.warning(f"Could not load existing data: {e}")

        LOGGER.info(
            f"Initialized JSONVectorStore: collection={collection_name}, "
            f"directory={persist_directory}, documents={len(self._documents)}"
        )

    @property
    def embeddings(self):
        """Get embeddings function for compatibility."""
        return self._embedding

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add texts to the vector store.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            ids: Optional list of IDs
            **kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Create documents
        new_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        # Generate embeddings
        new_embeddings = self._embedding.embed_documents(texts)
        new_embeddings_array = np.array(new_embeddings, dtype=np.float32)

        # Add to storage
        self._documents.extend(new_docs)
        self._document_ids.extend(ids)

        if self._embeddings is None:
            self._embeddings = new_embeddings_array
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings_array])

        LOGGER.info(f"Added {len(texts)} documents (total: {len(self._documents)})")

        return ids

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects
            ids: Optional list of IDs
            **kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, ids, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of similar documents
        """
        if not self._documents:
            return []

        # Get query embedding
        query_embedding = self._embedding.embed_query(query)
        query_embedding_array = np.array(query_embedding, dtype=np.float32)

        # Calculate cosine similarity
        scores = self._cosine_similarity(query_embedding_array, self._embeddings)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Return documents
        return [self._documents[i] for i in top_k_indices]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores.

        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of (document, score) tuples
        """
        if not self._documents:
            return []

        # Get query embedding
        query_embedding = self._embedding.embed_query(query)
        query_embedding_array = np.array(query_embedding, dtype=np.float32)

        # Calculate cosine similarity
        scores = self._cosine_similarity(query_embedding_array, self._embeddings)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Return documents with scores
        return [(self._documents[i], float(scores[i])) for i in top_k_indices]

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and documents.

        Args:
            query_vec: Query embedding vector (1D array)
            doc_vecs: Document embedding vectors (2D array)

        Returns:
            Similarity scores (1D array)
        """
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)

        # Calculate dot product (cosine similarity for normalized vectors)
        similarities = np.dot(doc_norms, query_norm)

        return similarities

    def persist(self):
        """Save vector store to disk."""
        if self.persist_directory is None:
            raise ValueError("persist_directory not set")

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Save documents
        documents_path = self.persist_directory / "documents.json"
        documents_data = [
            {
                'id': doc_id,
                'page_content': doc.page_content,
                'metadata': doc.metadata,
            }
            for doc_id, doc in zip(self._document_ids, self._documents)
        ]

        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, indent=2, ensure_ascii=False)

        # Save embeddings
        if self._embeddings is not None:
            embeddings_path = self.persist_directory / "embeddings.npy"
            np.save(embeddings_path, self._embeddings)

        # Save index metadata
        index_path = self.persist_directory / "index.json"
        index_data = {
            'collection_name': self.collection_name,
            'num_documents': len(self._documents),
            'embedding_dimensions': self._embeddings.shape[1] if self._embeddings is not None else 0,
            'version': '1.0',
        }

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)

        LOGGER.info(
            f"Persisted {len(self._documents)} documents to {self.persist_directory}"
        )

    def _load_from_disk(self):
        """Load vector store from disk."""
        if self.persist_directory is None or not self.persist_directory.exists():
            return

        # Load documents
        documents_path = self.persist_directory / "documents.json"
        if documents_path.exists():
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)

            self._documents = [
                Document(
                    page_content=doc['page_content'],
                    metadata=doc['metadata']
                )
                for doc in documents_data
            ]

            self._document_ids = [doc['id'] for doc in documents_data]

        # Load embeddings
        embeddings_path = self.persist_directory / "embeddings.npy"
        if embeddings_path.exists():
            self._embeddings = np.load(embeddings_path)

        LOGGER.info(f"Loaded {len(self._documents)} documents from {self.persist_directory}")

    @classmethod
    def load(
        cls,
        persist_directory: str,
        embedding,
        collection_name: str = "default",
    ) -> "JSONVectorStore":
        """
        Load vector store from disk.

        Args:
            persist_directory: Directory containing saved data
            embedding: Embeddings instance
            collection_name: Name of the collection

        Returns:
            Loaded JSONVectorStore
        """
        store = cls(
            embedding=embedding,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        return store

    def delete(self, ids: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        if ids is None:
            return False

        # Find indices to delete
        indices_to_delete = []
        for i, doc_id in enumerate(self._document_ids):
            if doc_id in ids:
                indices_to_delete.append(i)

        if not indices_to_delete:
            return False

        # Remove documents
        for index in sorted(indices_to_delete, reverse=True):
            del self._documents[index]
            del self._document_ids[index]

        # Remove embeddings
        if self._embeddings is not None:
            mask = np.ones(len(self._embeddings), dtype=bool)
            mask[indices_to_delete] = False
            self._embeddings = self._embeddings[mask]

        LOGGER.info(f"Deleted {len(indices_to_delete)} documents")

        return True

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return {
            'collection_name': self.collection_name,
            'num_documents': len(self._documents),
            'embedding_dimensions': self._embeddings.shape[1] if self._embeddings is not None else 0,
            'persist_directory': str(self.persist_directory) if self.persist_directory else None,
            'storage_type': 'json',
        }

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding,
        metadatas: Optional[List[Dict]] = None,
        persist_directory: Optional[str] = None,
        **kwargs
    ) -> "JSONVectorStore":
        """
        Create vector store from texts.

        Args:
            texts: List of text strings
            embedding: Embeddings instance
            metadatas: Optional list of metadata dicts
            persist_directory: Directory to save files
            **kwargs: Additional arguments

        Returns:
            New JSONVectorStore
        """
        store = cls(
            embedding=embedding,
            persist_directory=persist_directory,
        )
        store.add_texts(texts, metadatas, **kwargs)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding,
        persist_directory: Optional[str] = None,
        **kwargs
    ) -> "JSONVectorStore":
        """
        Create vector store from documents.

        Args:
            documents: List of Document objects
            embedding: Embeddings instance
            persist_directory: Directory to save files
            **kwargs: Additional arguments

        Returns:
            New JSONVectorStore
        """
        store = cls(
            embedding=embedding,
            persist_directory=persist_directory,
        )
        store.add_documents(documents, **kwargs)
        return store

    def export_to_json(self, output_path: str):
        """
        Export vector store to a single JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare export data
        export_data = {
            'collection_name': self.collection_name,
            'num_documents': len(self._documents),
            'documents': [
                {
                    'id': doc_id,
                    'page_content': doc.page_content,
                    'metadata': doc.metadata,
                    'embedding': self._embeddings[i].tolist() if self._embeddings is not None else None,
                }
                for i, (doc_id, doc) in enumerate(zip(self._document_ids, self._documents))
            ],
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Exported {len(self._documents)} documents to {output_path}")

    @classmethod
    def import_from_json(
        cls,
        input_path: str,
        embedding,
        persist_directory: Optional[str] = None,
    ) -> "JSONVectorStore":
        """
        Import vector store from a JSON file.

        Args:
            input_path: Path to input JSON file
            embedding: Embeddings instance
            persist_directory: Directory to save imported data

        Returns:
            New JSONVectorStore with imported data
        """
        input_path = Path(input_path)

        with open(input_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)

        store = cls(
            embedding=embedding,
            persist_directory=persist_directory,
            collection_name=import_data.get('collection_name', 'default'),
        )

        # Import documents
        for doc_data in import_data['documents']:
            doc = Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            )
            store._documents.append(doc)
            store._document_ids.append(doc_data['id'])

            # Import embedding if available
            if doc_data.get('embedding'):
                embedding_array = np.array(doc_data['embedding'], dtype=np.float32)
                if store._embeddings is None:
                    store._embeddings = embedding_array.reshape(1, -1)
                else:
                    store._embeddings = np.vstack([store._embeddings, embedding_array])

        LOGGER.info(f"Imported {len(store._documents)} documents from {input_path}")

        return store
