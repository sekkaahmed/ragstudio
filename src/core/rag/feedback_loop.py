"""
Feedback Loop for Atlas-RAG.

Collects user feedback on RAG responses and uses it to improve:
1. Retrieval quality (learn which documents are relevant)
2. Response quality (learn which answers are helpful)
3. Model performance tracking
4. Active learning for fine-tuning

Feedback types:
- Explicit: User ratings (thumbs up/down, 1-5 stars)
- Implicit: Click-through rate, time spent reading
- Corrections: User edits to responses

Benefits:
- Continuous improvement of RAG system
- Identify problematic queries/documents
- Guide retraining and fine-tuning
- A/B testing of different strategies
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

LOGGER = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback."""

    # User explicitly rated the response
    EXPLICIT_RATING = "explicit_rating"

    # User clicked on a source document
    CLICK_THROUGH = "click_through"

    # User spent time reading the response
    DWELL_TIME = "dwell_time"

    # User edited/corrected the response
    CORRECTION = "correction"

    # User flagged response as incorrect/harmful
    FLAG = "flag"


class FeedbackScore(str, Enum):
    """Feedback scores for explicit ratings."""

    POSITIVE = "positive"  # Thumbs up, helpful
    NEGATIVE = "negative"  # Thumbs down, not helpful
    NEUTRAL = "neutral"    # No strong opinion


@dataclass
class RAGFeedback:
    """
    Feedback entry for a RAG interaction.

    Tracks user feedback on a specific query-response pair.
    """

    # Unique feedback ID
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Query information
    query: str = ""
    query_id: Optional[str] = None

    # Response information
    response: str = ""
    response_id: Optional[str] = None

    # Source documents used
    source_document_ids: List[str] = field(default_factory=list)

    # Feedback type and score
    feedback_type: FeedbackType = FeedbackType.EXPLICIT_RATING
    score: Optional[FeedbackScore] = None

    # Numeric rating (e.g., 1-5 stars)
    numeric_rating: Optional[float] = None

    # User comments
    comment: Optional[str] = None

    # Click-through information
    clicked_documents: List[str] = field(default_factory=list)

    # Dwell time (seconds)
    dwell_time_seconds: Optional[float] = None

    # User correction (if provided)
    corrected_response: Optional[str] = None

    # Flag reason (if flagged)
    flag_reason: Optional[str] = None

    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # RAG configuration used
    retrieval_strategy: Optional[str] = None
    model_name: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert enums to strings
        if isinstance(data['feedback_type'], FeedbackType):
            data['feedback_type'] = data['feedback_type'].value
        if isinstance(data['score'], FeedbackScore):
            data['score'] = data['score'].value
        # Convert datetime to ISO string
        if isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data


class FeedbackCollector:
    """
    Collects and stores RAG feedback.

    Can be integrated with:
    - Metadata store (PostgreSQL)
    - Analytics platform
    - ML training pipeline

    Example:
        >>> from src.core.rag.feedback_loop import FeedbackCollector, FeedbackScore
        >>>
        >>> collector = FeedbackCollector(metadata_store=store)
        >>>
        >>> # User rates response
        >>> collector.record_rating(
        ...     query="What is machine learning?",
        ...     response="Machine learning is...",
        ...     score=FeedbackScore.POSITIVE,
        ...     user_id="user123",
        ... )
        >>>
        >>> # Get analytics
        >>> stats = collector.get_statistics()
        >>> print(f"Positive rate: {stats['positive_rate']}")
    """

    def __init__(
        self,
        metadata_store=None,
        storage_backend: Literal['memory', 'database', 'file'] = 'memory',
        storage_path: Optional[str] = None,
    ):
        """
        Initialize feedback collector.

        Args:
            metadata_store: MetadataStore instance for database storage
            storage_backend: Where to store feedback ('memory', 'database', 'file')
            storage_path: Path for file storage
        """
        self.metadata_store = metadata_store
        self.storage_backend = storage_backend
        self.storage_path = storage_path

        # In-memory storage
        self._feedback_entries: List[RAGFeedback] = []

        LOGGER.info(f"Initialized FeedbackCollector: backend={storage_backend}")

    def record_rating(
        self,
        query: str,
        response: str,
        score: FeedbackScore,
        source_document_ids: Optional[List[str]] = None,
        numeric_rating: Optional[float] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ) -> RAGFeedback:
        """
        Record explicit user rating.

        Args:
            query: User query
            response: RAG response
            score: Feedback score (positive/negative/neutral)
            source_document_ids: IDs of source documents
            numeric_rating: Numeric rating (e.g., 1-5)
            comment: User comment
            user_id: User ID
            **kwargs: Additional metadata

        Returns:
            Created feedback entry
        """
        feedback = RAGFeedback(
            query=query,
            response=response,
            source_document_ids=source_document_ids or [],
            feedback_type=FeedbackType.EXPLICIT_RATING,
            score=score,
            numeric_rating=numeric_rating,
            comment=comment,
            user_id=user_id,
            metadata=kwargs,
        )

        self._store_feedback(feedback)

        LOGGER.info(
            f"Recorded rating: query='{query[:50]}...', score={score.value}"
        )

        return feedback

    def record_click_through(
        self,
        query: str,
        clicked_document_ids: List[str],
        source_document_ids: List[str],
        user_id: Optional[str] = None,
        **kwargs
    ) -> RAGFeedback:
        """
        Record click-through on source documents.

        Args:
            query: User query
            clicked_document_ids: Documents user clicked on
            source_document_ids: All source documents shown
            user_id: User ID
            **kwargs: Additional metadata

        Returns:
            Created feedback entry
        """
        feedback = RAGFeedback(
            query=query,
            source_document_ids=source_document_ids,
            clicked_documents=clicked_document_ids,
            feedback_type=FeedbackType.CLICK_THROUGH,
            user_id=user_id,
            metadata=kwargs,
        )

        self._store_feedback(feedback)

        LOGGER.debug(
            f"Recorded click-through: {len(clicked_document_ids)} documents clicked"
        )

        return feedback

    def record_dwell_time(
        self,
        query: str,
        response: str,
        dwell_time_seconds: float,
        user_id: Optional[str] = None,
        **kwargs
    ) -> RAGFeedback:
        """
        Record time user spent reading response.

        Args:
            query: User query
            response: RAG response
            dwell_time_seconds: Time spent (seconds)
            user_id: User ID
            **kwargs: Additional metadata

        Returns:
            Created feedback entry
        """
        feedback = RAGFeedback(
            query=query,
            response=response,
            feedback_type=FeedbackType.DWELL_TIME,
            dwell_time_seconds=dwell_time_seconds,
            user_id=user_id,
            metadata=kwargs,
        )

        self._store_feedback(feedback)

        return feedback

    def record_correction(
        self,
        query: str,
        original_response: str,
        corrected_response: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> RAGFeedback:
        """
        Record user correction to response.

        Args:
            query: User query
            original_response: Original RAG response
            corrected_response: User's corrected version
            user_id: User ID
            **kwargs: Additional metadata

        Returns:
            Created feedback entry
        """
        feedback = RAGFeedback(
            query=query,
            response=original_response,
            corrected_response=corrected_response,
            feedback_type=FeedbackType.CORRECTION,
            user_id=user_id,
            metadata=kwargs,
        )

        self._store_feedback(feedback)

        LOGGER.info(
            f"Recorded correction: query='{query[:50]}...'"
        )

        return feedback

    def record_flag(
        self,
        query: str,
        response: str,
        flag_reason: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> RAGFeedback:
        """
        Record user flag (incorrect/harmful response).

        Args:
            query: User query
            response: RAG response
            flag_reason: Reason for flag
            user_id: User ID
            **kwargs: Additional metadata

        Returns:
            Created feedback entry
        """
        feedback = RAGFeedback(
            query=query,
            response=response,
            feedback_type=FeedbackType.FLAG,
            flag_reason=flag_reason,
            score=FeedbackScore.NEGATIVE,
            user_id=user_id,
            metadata=kwargs,
        )

        self._store_feedback(feedback)

        LOGGER.warning(
            f"Recorded flag: query='{query[:50]}...', reason={flag_reason}"
        )

        return feedback

    def _store_feedback(self, feedback: RAGFeedback):
        """Store feedback based on storage backend."""
        # Always store in memory
        self._feedback_entries.append(feedback)

        # Store in database if available
        if self.storage_backend == 'database' and self.metadata_store:
            try:
                self.metadata_store.create_audit_log(
                    resource_type='rag_feedback',
                    resource_id=feedback.feedback_id,
                    action=feedback.feedback_type.value,
                    metadata=feedback.to_dict(),
                )
            except Exception as e:
                LOGGER.error(f"Error storing feedback in database: {e}")

        # Store in file if requested
        if self.storage_backend == 'file' and self.storage_path:
            try:
                import json
                from pathlib import Path

                filepath = Path(self.storage_path) / f"{feedback.feedback_id}.json"
                filepath.parent.mkdir(parents=True, exist_ok=True)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(feedback.to_dict(), f, indent=2, ensure_ascii=False)

            except Exception as e:
                LOGGER.error(f"Error storing feedback in file: {e}")

    def get_statistics(
        self,
        feedback_type: Optional[FeedbackType] = None,
        time_window_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get feedback statistics.

        Args:
            feedback_type: Filter by feedback type
            time_window_hours: Only include feedback from last N hours

        Returns:
            Statistics dictionary
        """
        # Filter entries
        entries = self._feedback_entries

        if feedback_type:
            entries = [e for e in entries if e.feedback_type == feedback_type]

        if time_window_hours:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
            entries = [e for e in entries if e.timestamp >= cutoff]

        if not entries:
            return {'total': 0}

        # Calculate stats
        total = len(entries)
        positive = sum(1 for e in entries if e.score == FeedbackScore.POSITIVE)
        negative = sum(1 for e in entries if e.score == FeedbackScore.NEGATIVE)
        neutral = sum(1 for e in entries if e.score == FeedbackScore.NEUTRAL)

        numeric_ratings = [
            e.numeric_rating for e in entries
            if e.numeric_rating is not None
        ]

        avg_numeric_rating = (
            sum(numeric_ratings) / len(numeric_ratings)
            if numeric_ratings else None
        )

        flags = sum(1 for e in entries if e.feedback_type == FeedbackType.FLAG)

        return {
            'total': total,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_rate': positive / total if total > 0 else 0,
            'negative_rate': negative / total if total > 0 else 0,
            'avg_numeric_rating': avg_numeric_rating,
            'flags': flags,
            'flag_rate': flags / total if total > 0 else 0,
        }

    def get_problematic_queries(
        self,
        min_negative_rate: float = 0.5,
        min_samples: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Identify queries with poor feedback.

        Args:
            min_negative_rate: Minimum negative rate to consider problematic
            min_samples: Minimum number of feedback samples required

        Returns:
            List of problematic query info
        """
        # Group by query
        query_feedback: Dict[str, List[RAGFeedback]] = {}

        for entry in self._feedback_entries:
            if entry.query not in query_feedback:
                query_feedback[entry.query] = []
            query_feedback[entry.query].append(entry)

        # Find problematic
        problematic = []

        for query, entries in query_feedback.items():
            if len(entries) < min_samples:
                continue

            negative_count = sum(
                1 for e in entries if e.score == FeedbackScore.NEGATIVE
            )
            negative_rate = negative_count / len(entries)

            if negative_rate >= min_negative_rate:
                problematic.append({
                    'query': query,
                    'total_feedback': len(entries),
                    'negative_rate': negative_rate,
                    'negative_count': negative_count,
                })

        # Sort by negative rate
        problematic.sort(key=lambda x: x['negative_rate'], reverse=True)

        return problematic

    def export_feedback(
        self,
        filepath: str,
        format: Literal['json', 'csv'] = 'json',
    ):
        """
        Export all feedback to file.

        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
        """
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                data = [entry.to_dict() for entry in self._feedback_entries]
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == 'csv':
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if not self._feedback_entries:
                    return

                writer = csv.DictWriter(f, fieldnames=self._feedback_entries[0].to_dict().keys())
                writer.writeheader()
                for entry in self._feedback_entries:
                    writer.writerow(entry.to_dict())

        LOGGER.info(f"Exported {len(self._feedback_entries)} feedback entries to {filepath}")


def create_feedback_collector(
    metadata_store=None,
    storage_backend: str = 'memory',
    **kwargs
) -> FeedbackCollector:
    """
    Helper function to create feedback collector.

    Args:
        metadata_store: MetadataStore for database storage
        storage_backend: Storage backend ('memory', 'database', 'file')
        **kwargs: Additional parameters

    Returns:
        Configured FeedbackCollector

    Example:
        >>> collector = create_feedback_collector(
        ...     metadata_store=store,
        ...     storage_backend='database',
        ... )
    """
    return FeedbackCollector(
        metadata_store=metadata_store,
        storage_backend=storage_backend,
        **kwargs
    )
