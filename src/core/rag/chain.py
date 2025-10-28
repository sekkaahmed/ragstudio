"""
RAG Chain for Atlas-RAG.

Complete Retrieval-Augmented Generation pipeline using LangChain.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
import time

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.core.rag.retrievers import AtlasRetriever
from src.core.storage.metadata_store import MetadataStore

LOGGER = logging.getLogger(__name__)


# Default RAG prompt template
DEFAULT_RAG_PROMPT = """Utilise les informations suivantes pour répondre à la question.
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas, n'essaie pas d'inventer une réponse.

Contexte:
{context}

Question: {question}

Réponse détaillée:"""


class AtlasRAGChain:
    """
    Complete RAG chain for Atlas-RAG.

    Features:
    - Question answering with context
    - Source attribution
    - Metadata tracking
    - Multiple LLM support
    """

    def __init__(
        self,
        retriever: AtlasRetriever,
        llm,
        metadata_store: Optional[MetadataStore] = None,
        prompt_template: Optional[str] = None,
        return_source_documents: bool = True,
    ):
        """
        Initialize RAG chain.

        Args:
            retriever: AtlasRetriever instance
            llm: Language model (OpenAI, HuggingFace, etc.)
            metadata_store: Optional MetadataStore for logging
            prompt_template: Custom prompt template
            return_source_documents: Include source documents in response
        """
        self.retriever = retriever
        self.llm = llm
        self.metadata_store = metadata_store
        self.return_source_documents = return_source_documents

        # Create prompt
        self.prompt = PromptTemplate(
            template=prompt_template or DEFAULT_RAG_PROMPT,
            input_variables=["context", "question"]
        )

        # Create chain
        self._chain = None

        LOGGER.info("Initialized AtlasRAGChain")

    @property
    def chain(self) -> RetrievalQA:
        """Lazy-load RAG chain."""
        if self._chain is None:
            base_retriever = self.retriever.get_basic_retriever()

            self._chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=base_retriever,
                return_source_documents=self.return_source_documents,
                chain_type_kwargs={"prompt": self.prompt}
            )

        return self._chain

    def query(
        self,
        question: str,
        retrieval_strategy: str = "basic",
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User question
            retrieval_strategy: Retrieval strategy to use

        Returns:
            Dictionary with:
            {
                'query': str,
                'result': str,
                'source_documents': List[Document],
                'retrieval_time': float,
                'generation_time': float,
                'total_time': float,
            }
        """
        start_time = time.time()

        LOGGER.info(f"RAG Query: {question[:100]}...")

        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()

        source_docs = self.retriever.retrieve(
            query=question,
            strategy=retrieval_strategy,
            llm=self.llm if retrieval_strategy != "basic" else None,
        )

        retrieval_time = time.time() - retrieval_start

        LOGGER.info(f"✓ Retrieved {len(source_docs)} documents in {retrieval_time:.2f}s")

        # Step 2: Generate answer
        generation_start = time.time()

        response = self.chain.invoke({"query": question})

        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        LOGGER.info(f"✓ Generated answer in {generation_time:.2f}s (total: {total_time:.2f}s)")

        # Format response
        result = {
            'query': question,
            'result': response.get('result', ''),
            'source_documents': response.get('source_documents', source_docs),
            'retrieval_time': round(retrieval_time, 2),
            'generation_time': round(generation_time, 2),
            'total_time': round(total_time, 2),
            'num_sources': len(source_docs),
        }

        # Log to metadata store (optional)
        if self.metadata_store:
            try:
                self.metadata_store.create_audit_log(
                    resource_type='rag_query',
                    resource_id=str(time.time()),  # Use timestamp as ID
                    action='query',
                    user_id='system',
                    metadata={
                        'question': question[:200],
                        'answer': response.get('result', '')[:200],
                        'num_sources': len(source_docs),
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                    }
                )
            except Exception as e:
                LOGGER.warning(f"Failed to log to metadata store: {e}")

        return result

    def format_answer(self, result: Dict[str, Any]) -> str:
        """
        Format answer with sources for display.

        Args:
            result: Query result dictionary

        Returns:
            Formatted string with answer and sources
        """
        output = []

        # Answer
        output.append("=" * 80)
        output.append("RÉPONSE")
        output.append("=" * 80)
        output.append(result['result'])
        output.append("")

        # Sources
        if result.get('source_documents'):
            output.append("=" * 80)
            output.append("SOURCES")
            output.append("=" * 80)

            for i, doc in enumerate(result['source_documents'], 1):
                metadata = doc.metadata
                filename = metadata.get('source_name', 'Unknown')
                page = metadata.get('page', 'N/A')

                output.append(f"\n[{i}] {filename} (page {page})")
                preview = doc.page_content[:200].replace('\n', ' ')
                output.append(f"    {preview}...")

        # Stats
        output.append("")
        output.append("=" * 80)
        output.append("STATISTIQUES")
        output.append("=" * 80)
        output.append(f"Retrieval time: {result['retrieval_time']}s")
        output.append(f"Generation time: {result['generation_time']}s")
        output.append(f"Total time: {result['total_time']}s")
        output.append(f"Sources used: {result['num_sources']}")

        return "\n".join(output)


def create_rag_chain(
    retriever: AtlasRetriever,
    llm,
    **kwargs
) -> AtlasRAGChain:
    """
    Create RAG chain.

    Args:
        retriever: AtlasRetriever instance
        llm: Language model
        **kwargs: Additional arguments

    Returns:
        AtlasRAGChain instance
    """
    return AtlasRAGChain(
        retriever=retriever,
        llm=llm,
        **kwargs
    )
