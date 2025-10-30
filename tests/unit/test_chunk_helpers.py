"""
Tests unitaires pour les fonctions helpers de chunk.py.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime

from src.core.cli.commands.chunk import (
    _generate_processing_summary,
    Document,
)


@pytest.fixture
def mock_config():
    """Mock AtlasConfig for testing."""
    config = Mock()

    # LLM config
    config.llm = Mock()
    config.llm.use_llm = True
    config.llm.provider = "ollama"
    config.llm.model = "mistral:latest"
    config.llm.is_local = True

    # OCR config
    config.ocr = Mock()
    config.ocr.use_advanced_ocr = True
    config.ocr.dictionary_threshold = 0.30
    config.ocr.dynamic_threshold = True
    config.ocr.enable_fallback = True

    # Chunking config
    config.chunking = Mock()
    config.chunking.strategy = "semantic"
    config.chunking.max_tokens = 400
    config.chunking.overlap = 50

    return config


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        Mock(text="This is chunk 1 with some text content."),
        Mock(text="This is chunk 2 with different text."),
        Mock(text="This is chunk 3."),
    ]


@pytest.fixture
def sample_file_path(tmp_path):
    """Create a sample file for testing."""
    file_path = tmp_path / "test.pdf"
    file_path.write_text("test content")
    return file_path


class TestGenerateProcessingSummary:
    """Tests for _generate_processing_summary function."""

    def test_basic_summary(self, mock_config, sample_chunks, sample_file_path):
        """Test basic summary generation."""
        processing_data = {
            "text_length": 1234,
            "language": "en",
            "total_time": 5.5
        }

        summary = _generate_processing_summary(
            file_path=sample_file_path,
            config=mock_config,
            processing_data=processing_data,
            chunks=sample_chunks,
            success=True
        )

        # Check metadata section
        assert "metadata" in summary
        assert summary["metadata"]["success"] is True
        assert summary["metadata"]["atlas_rag_version"] == "1.0.0"
        assert "processing_timestamp" in summary["metadata"]
        assert summary["metadata"]["errors"] == []

        # Check document section
        assert "document" in summary
        assert summary["document"]["filename"] == "test.pdf"
        assert summary["document"]["format"] == ".pdf"
        assert summary["document"]["text_length"] == 1234
        assert summary["document"]["language"] == "en"

        # Check configuration section
        assert "configuration" in summary
        assert summary["configuration"]["llm"]["enabled"] is True
        assert summary["configuration"]["llm"]["provider"] == "ollama"
        assert summary["configuration"]["ocr"]["advanced_ocr_enabled"] is True
        assert summary["configuration"]["chunking"]["strategy"] == "semantic"

        # Check processing section
        assert "processing" in summary
        assert summary["processing"]["total_time_seconds"] == 5.5

        # Check results section
        assert "results" in summary
        assert summary["results"]["chunks"]["total_count"] == 3
        assert summary["results"]["chunks"]["total_text_length"] > 0

    def test_summary_with_ocr_data(self, mock_config, sample_chunks, sample_file_path):
        """Test summary generation with OCR data."""
        processing_data = {
            "text_length": 1234,
            "language": "en",
            "total_time": 10.0,
            "ocr_time": 3.5,
            "ocr_result": {
                "metadata": {
                    "ocr_engine": "qwen-vl",
                    "success": True,
                    "quality_metrics": {"confidence": 0.95},
                    "fallback_from": None,
                    "fallback_reason": None
                },
                "routing_decisions": [
                    {"step": "ocr_quality_detection", "quality": "LOW"}
                ]
            }
        }

        summary = _generate_processing_summary(
            file_path=sample_file_path,
            config=mock_config,
            processing_data=processing_data,
            chunks=sample_chunks
        )

        # Check OCR stage in processing
        assert "ocr" in summary["processing"]["stages"]
        ocr_stage = summary["processing"]["stages"]["ocr"]

        assert ocr_stage["time_seconds"] == 3.5
        assert ocr_stage["engine"] == "qwen-vl"
        assert ocr_stage["success"] is True
        assert ocr_stage["fallback_used"] is False
        assert ocr_stage["quality_metrics"] == {"confidence": 0.95}
        assert len(ocr_stage["routing_decisions"]) == 1

    def test_summary_with_ocr_fallback(self, mock_config, sample_chunks, sample_file_path):
        """Test summary with OCR fallback scenario."""
        processing_data = {
            "text_length": 1234,
            "language": "en",
            "total_time": 12.0,
            "ocr_time": 5.0,
            "ocr_result": {
                "metadata": {
                    "ocr_engine": "classic",
                    "success": True,
                    "quality_metrics": {},
                    "fallback_from": "qwen-vl",
                    "fallback_reason": "Qwen-VL unavailable"
                },
                "routing_decisions": []
            }
        }

        summary = _generate_processing_summary(
            file_path=sample_file_path,
            config=mock_config,
            processing_data=processing_data,
            chunks=sample_chunks
        )

        # Check fallback information
        ocr_stage = summary["processing"]["stages"]["ocr"]
        assert ocr_stage["fallback_used"] is True
        assert ocr_stage["fallback_reason"] == "Qwen-VL unavailable"

    def test_summary_with_strategy_selection(self, mock_config, sample_chunks, sample_file_path):
        """Test summary with strategy selection data."""
        processing_data = {
            "text_length": 1234,
            "language": "en",
            "total_time": 8.0,
            "strategy_selection": {
                "predicted_strategy": "semantic",
                "confidence": 0.92,
                "features": {"complexity": 0.7}
            }
        }

        summary = _generate_processing_summary(
            file_path=sample_file_path,
            config=mock_config,
            processing_data=processing_data,
            chunks=sample_chunks
        )

        # Check strategy selection stage
        assert "strategy_selection" in summary["processing"]["stages"]
        strategy_stage = summary["processing"]["stages"]["strategy_selection"]

        assert strategy_stage["predicted_strategy"] == "semantic"
        assert strategy_stage["confidence"] == 0.92

    def test_summary_with_errors(self, mock_config, sample_chunks, sample_file_path):
        """Test summary generation with errors."""
        processing_data = {
            "text_length": 0,
            "language": "unknown",
            "total_time": 2.0
        }
        errors = ["OCR failed", "Extraction incomplete"]

        summary = _generate_processing_summary(
            file_path=sample_file_path,
            config=mock_config,
            processing_data=processing_data,
            chunks=sample_chunks,
            success=False,
            errors=errors
        )

        # Check error information
        assert summary["metadata"]["success"] is False
        assert summary["metadata"]["errors"] == errors

    def test_summary_empty_chunks(self, mock_config, sample_file_path):
        """Test summary with no chunks."""
        processing_data = {
            "text_length": 0,
            "language": "unknown",
            "total_time": 1.0
        }

        summary = _generate_processing_summary(
            file_path=sample_file_path,
            config=mock_config,
            processing_data=processing_data,
            chunks=[],
            success=True
        )

        # Check chunks statistics with empty list
        chunks_info = summary["results"]["chunks"]
        assert chunks_info["total_count"] == 0
        assert chunks_info["average_size_chars"] == 0
        assert chunks_info["min_size_chars"] == 0
        assert chunks_info["max_size_chars"] == 0
        assert chunks_info["total_text_length"] == 0

    def test_summary_chunk_statistics(self, mock_config, sample_file_path):
        """Test chunk statistics calculation."""
        chunks = [
            Mock(text="a" * 100),  # 100 chars
            Mock(text="b" * 200),  # 200 chars
            Mock(text="c" * 50),   # 50 chars
        ]
        processing_data = {
            "text_length": 350,
            "language": "en",
            "total_time": 3.0
        }

        summary = _generate_processing_summary(
            file_path=sample_file_path,
            config=mock_config,
            processing_data=processing_data,
            chunks=chunks
        )

        chunks_info = summary["results"]["chunks"]
        assert chunks_info["total_count"] == 3
        assert chunks_info["average_size_chars"] == 116  # (100+200+50)//3
        assert chunks_info["min_size_chars"] == 50
        assert chunks_info["max_size_chars"] == 200
        assert chunks_info["total_text_length"] == 350

    def test_summary_chunking_time(self, mock_config, sample_chunks, sample_file_path):
        """Test chunking time included in summary."""
        processing_data = {
            "text_length": 1234,
            "language": "en",
            "total_time": 5.0,
            "chunking_time": 2.5
        }

        summary = _generate_processing_summary(
            file_path=sample_file_path,
            config=mock_config,
            processing_data=processing_data,
            chunks=sample_chunks
        )

        # Check chunking stage timing
        assert "chunking" in summary["processing"]["stages"]
        assert summary["processing"]["stages"]["chunking"]["time_seconds"] == 2.5


class TestDocumentClass:
    """Tests for Document class."""

    def test_document_creation(self):
        """Test document creation with all fields."""
        doc = Document(
            text="Sample text content",
            metadata={"key": "value"},
            source_path="/path/to/file.pdf",
            id="doc123"
        )

        assert doc.text == "Sample text content"
        assert doc.metadata == {"key": "value"}
        assert doc.source_path == "/path/to/file.pdf"
        assert doc.id == "doc123"

    def test_document_without_id(self):
        """Test document creation without ID."""
        doc = Document(
            text="Sample text",
            metadata={},
            source_path="/path/to/file.pdf"
        )

        assert doc.text == "Sample text"
        assert doc.id is None

    def test_document_empty_metadata(self):
        """Test document with empty metadata."""
        doc = Document(
            text="Sample text",
            metadata={},
            source_path="/path/to/file.pdf"
        )

        assert doc.metadata == {}

    def test_document_complex_metadata(self):
        """Test document with complex metadata."""
        metadata = {
            "source": "test.pdf",
            "page": 1,
            "tags": ["important", "draft"],
            "nested": {"key": "value"}
        }

        doc = Document(
            text="Sample text",
            metadata=metadata,
            source_path="/path/to/file.pdf"
        )

        assert doc.metadata == metadata
        assert doc.metadata["tags"] == ["important", "draft"]
        assert doc.metadata["nested"]["key"] == "value"
