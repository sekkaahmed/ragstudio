"""
Query Expansion for Atlas-RAG.

Expands user queries to improve retrieval recall by:
1. Generating synonyms and related terms
2. Reformulating questions in different ways
3. Adding context from conversation history
4. Handling multilingual queries

Benefits:
- Better recall: Captures semantically similar documents with different wording
- Handles vocabulary mismatch: User words may differ from document terminology
- Improves robustness: Multiple query variants increase chance of good matches
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass
from enum import Enum

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

LOGGER = logging.getLogger(__name__)


class ExpansionStrategy(str, Enum):
    """Query expansion strategies."""

    # Generate multiple reformulations of the query
    MULTI_QUERY = "multi_query"

    # Add synonyms using WordNet or similar
    SYNONYMS = "synonyms"

    # Use LLM to generate related questions
    LLM_GENERATED = "llm_generated"

    # Combine original + synonyms + reformulations
    HYBRID = "hybrid"


@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion."""

    # Expansion strategy
    strategy: ExpansionStrategy = ExpansionStrategy.MULTI_QUERY

    # Number of expanded queries to generate
    num_queries: int = 3

    # Whether to include original query
    include_original: bool = True

    # Maximum query length (characters)
    max_query_length: int = 500

    # Language for expansion (for synonym-based strategies)
    language: str = 'en'


class QueryExpander:
    """
    Expands queries to improve retrieval coverage.

    Supports multiple expansion strategies:
    - Multi-query: Generate reformulations using LLM
    - Synonyms: Add synonyms for key terms
    - LLM-generated: Use LLM to create related questions
    - Hybrid: Combine multiple strategies

    Example:
        >>> from src.core.rag.query_expansion import QueryExpander
        >>>
        >>> expander = QueryExpander(llm=llm)
        >>> queries = expander.expand("What is machine learning?")
        >>> # ['What is machine learning?',
        >>> #  'Define machine learning',
        >>> #  'Explain the concept of machine learning']
    """

    def __init__(
        self,
        llm=None,
        config: Optional[QueryExpansionConfig] = None,
    ):
        """
        Initialize query expander.

        Args:
            llm: Language model for LLM-based expansion (optional)
            config: Query expansion configuration
        """
        self.llm = llm
        self.config = config or QueryExpansionConfig()

        # Check LLM availability for LLM-based strategies
        if self.config.strategy in [ExpansionStrategy.MULTI_QUERY, ExpansionStrategy.LLM_GENERATED, ExpansionStrategy.HYBRID]:
            if self.llm is None:
                LOGGER.warning(
                    f"LLM required for strategy '{self.config.strategy}' but not provided. "
                    "Falling back to SYNONYMS strategy."
                )
                self.config.strategy = ExpansionStrategy.SYNONYMS

        LOGGER.info(
            f"Initialized QueryExpander: strategy={self.config.strategy}, "
            f"num_queries={self.config.num_queries}"
        )

    def expand(self, query: str) -> List[str]:
        """
        Expand query into multiple variants.

        Args:
            query: Original query

        Returns:
            List of expanded queries (includes original if config.include_original=True)
        """
        if not query or not query.strip():
            return [query]

        # Truncate if too long
        query = query[:self.config.max_query_length]

        # Choose expansion strategy
        if self.config.strategy == ExpansionStrategy.MULTI_QUERY:
            expanded = self._expand_multi_query(query)
        elif self.config.strategy == ExpansionStrategy.SYNONYMS:
            expanded = self._expand_synonyms(query)
        elif self.config.strategy == ExpansionStrategy.LLM_GENERATED:
            expanded = self._expand_llm_generated(query)
        elif self.config.strategy == ExpansionStrategy.HYBRID:
            expanded = self._expand_hybrid(query)
        else:
            expanded = [query]

        # Add original query if requested
        if self.config.include_original and query not in expanded:
            expanded = [query] + expanded

        # Deduplicate while preserving order
        seen: Set[str] = set()
        deduplicated = []
        for q in expanded:
            q_clean = q.strip()
            if q_clean and q_clean.lower() not in seen:
                seen.add(q_clean.lower())
                deduplicated.append(q_clean)

        LOGGER.debug(
            f"Expanded query '{query[:50]}...' into {len(deduplicated)} variants"
        )

        return deduplicated

    def _expand_multi_query(self, query: str) -> List[str]:
        """
        Generate multiple reformulations using LLM.

        Creates semantically similar questions with different wording.
        """
        if self.llm is None:
            return [query]

        prompt = f"""You are a helpful AI assistant. Generate {self.config.num_queries} different versions of the following question. Each version should ask the same thing but use different words and phrasing.

Original question: {query}

Alternative versions (one per line):
"""

        try:
            response = self.llm.invoke(prompt)

            # Parse response (assuming one query per line)
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            queries = [
                line.strip().lstrip('123456789.-*) ')
                for line in response_text.strip().split('\n')
                if line.strip()
            ]

            # Limit to num_queries
            return queries[:self.config.num_queries]

        except Exception as e:
            LOGGER.error(f"Error in multi-query expansion: {e}")
            return [query]

    def _expand_synonyms(self, query: str) -> List[str]:
        """
        Expand query with synonyms for key terms.

        Uses simple word replacement with common synonyms.
        For production, consider using WordNet or custom thesaurus.
        """
        # Simple synonym dictionary (English)
        # In production, use proper NLP library like nltk.wordnet
        SYNONYMS = {
            'what is': ['define', 'explain', 'describe'],
            'how to': ['method to', 'way to', 'process of'],
            'machine learning': ['ML', 'artificial intelligence', 'AI'],
            'artificial intelligence': ['AI', 'machine learning', 'ML'],
            'data science': ['analytics', 'data analysis', 'statistical analysis'],
            'grammaire': ['règles grammaticales', 'syntaxe', 'structure grammaticale'],
            'français': ['langue française', 'francophone'],
        }

        expanded = [query]

        query_lower = query.lower()

        # Try to find and replace synonyms
        for term, synonyms in SYNONYMS.items():
            if term in query_lower:
                for synonym in synonyms[:self.config.num_queries - 1]:
                    # Case-preserving replacement
                    import re
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    new_query = pattern.sub(synonym, query, count=1)
                    if new_query != query:
                        expanded.append(new_query)

        return expanded[:self.config.num_queries]

    def _expand_llm_generated(self, query: str) -> List[str]:
        """
        Generate related questions using LLM.

        Creates questions that might retrieve relevant documents for the original query.
        """
        if self.llm is None:
            return [query]

        prompt = f"""You are a helpful AI assistant. Given the question below, generate {self.config.num_queries} related questions that someone might ask if they are interested in this topic. The questions should be different but related.

Original question: {query}

Related questions (one per line):
"""

        try:
            response = self.llm.invoke(prompt)

            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            queries = [
                line.strip().lstrip('123456789.-*) ')
                for line in response_text.strip().split('\n')
                if line.strip()
            ]

            return queries[:self.config.num_queries]

        except Exception as e:
            LOGGER.error(f"Error in LLM-generated expansion: {e}")
            return [query]

    def _expand_hybrid(self, query: str) -> List[str]:
        """
        Combine multiple expansion strategies.

        Uses both multi-query and synonym-based expansion.
        """
        # Get multi-query variants
        multi_queries = self._expand_multi_query(query)

        # Get synonym variants
        synonym_queries = self._expand_synonyms(query)

        # Combine
        combined = multi_queries + synonym_queries

        # Deduplicate
        seen = set()
        deduplicated = []
        for q in combined:
            if q.lower() not in seen:
                seen.add(q.lower())
                deduplicated.append(q)

        return deduplicated[:self.config.num_queries]


class QueryExpansionRetriever:
    """
    Retriever that uses query expansion to improve recall.

    Expands the query, retrieves documents for each variant,
    and merges results.

    Example:
        >>> from src.core.rag.query_expansion import QueryExpansionRetriever
        >>> from src.core.rag.vector_store import VectorStoreManager
        >>>
        >>> retriever = QueryExpansionRetriever(
        ...     vector_store_manager=vsm,
        ...     expander=expander,
        ...     k_per_query=3,
        ...     final_k=5,
        ... )
        >>>
        >>> results = retriever.retrieve("What is machine learning?")
    """

    def __init__(
        self,
        vector_store_manager,
        expander: QueryExpander,
        k_per_query: int = 3,
        final_k: int = 5,
    ):
        """
        Initialize query expansion retriever.

        Args:
            vector_store_manager: Vector store for retrieval
            expander: Query expander
            k_per_query: Number of documents to retrieve per expanded query
            final_k: Final number of documents to return
        """
        self.vector_store_manager = vector_store_manager
        self.expander = expander
        self.k_per_query = k_per_query
        self.final_k = final_k

        LOGGER.info(
            f"Initialized QueryExpansionRetriever: "
            f"k_per_query={k_per_query}, final_k={final_k}"
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents using query expansion.

        Args:
            query: Original query

        Returns:
            Merged and deduplicated documents
        """
        # Expand query
        expanded_queries = self.expander.expand(query)

        LOGGER.debug(f"Expanded into {len(expanded_queries)} queries")

        # Retrieve for each expanded query
        all_docs = []
        seen_content = set()

        for expanded_query in expanded_queries:
            docs = self.vector_store_manager.similarity_search(
                expanded_query,
                k=self.k_per_query,
            )

            # Deduplicate by content
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)

        # Return top final_k
        return all_docs[:self.final_k]


def create_query_expander(
    llm=None,
    strategy: str = "multi_query",
    num_queries: int = 3,
    include_original: bool = True,
    **kwargs
) -> QueryExpander:
    """
    Helper function to create query expander.

    Args:
        llm: Language model (required for LLM-based strategies)
        strategy: Expansion strategy ('multi_query', 'synonyms', 'llm_generated', 'hybrid')
        num_queries: Number of queries to generate
        include_original: Include original query in results
        **kwargs: Additional config parameters

    Returns:
        Configured QueryExpander

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-3.5-turbo")
        >>> expander = create_query_expander(
        ...     llm=llm,
        ...     strategy="multi_query",
        ...     num_queries=3,
        ... )
    """
    config = QueryExpansionConfig(
        strategy=ExpansionStrategy(strategy),
        num_queries=num_queries,
        include_original=include_original,
        **kwargs
    )

    return QueryExpander(llm=llm, config=config)
