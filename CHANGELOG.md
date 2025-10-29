# Changelog

All notable changes to RAG Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-10-29

### 🎨 Rebranding

**RAG Studio** - Professional rebrand with improved naming:
- **Project name**: Atlas-RAG → **RAG Studio**
- **Package name**: atlas-rag → **ragctl** (follows kubectl/systemctl convention)
- **CLI command**: `atlas-rag` → `ragctl`
- **Repository**: horiz-data/atlas-rag → horiz-data/ragstudio

### ⚡ Performance Improvements

**CLI Startup Optimization** - 100x faster:
- Implemented lazy imports for heavy modules (torch, transformers, langchain)
- Startup time: 5-10s → 0.08s
- `ragctl --version` now instant
- Commands load dependencies only when needed

### ✨ New Features

**User Experience**:
- Suppressed transformers FutureWarning for cleaner output
- Updated all help text and branding to RAG Studio
- Improved version display format

### 🔒 Security & CI/CD

**Comprehensive AI/ML Security Pipeline**:
- Added 12 security tools across 10 jobs
- **Secrets detection**: Gitleaks, TruffleHog (AI API keys: OpenAI, Anthropic, HuggingFace)
- **SAST**: Bandit, Semgrep (AI/ML-specific patterns)
- **Dependencies**: Safety, Pip-audit, Snyk
- **Supply chain**: SBOM generation (CycloneDX), Dependency Review
- **License compliance**: pip-licenses (GPL/AGPL blocking)
- **Vulnerabilities**: Trivy, CodeQL

**CI/CD Workflows**:
- `pr-validation.yml`: Security validation for contributors (no deployment)
- `release-deploy.yml`: Admin-only PyPI deployment with Trusted Publisher
- `.gitleaks.toml`: Custom AI/ML API key detection patterns
- `SECURITY.md`: Complete security documentation

### 📝 Documentation

- Updated all documentation to RAG Studio branding
- Updated all CLI examples to use `ragctl` command
- Added PyPI installation instructions
- All GitHub URLs updated to ragstudio repository
- Updated version references to 0.1.2

### 🔧 Migration Guide

If you installed version 0.1.0 or 0.1.1:

```bash
# Uninstall old package
pip uninstall atlas-rag

# Install new package
pip install ragctl

# Update your commands
atlas-rag → ragctl
```

**Breaking changes**:
- CLI command changed from `atlas-rag` to `ragctl`
- Package name changed from `atlas-rag` to `ragctl`
- Configuration directory remains `~/.atlasrag/` (no change)

---

## [0.1.0] - 2025-01-28

### 🎉 Initial Public Release

First open-source release of Atlas-RAG - A production-ready document processing CLI for RAG applications.

---

## What's Included

### Core CLI Commands

#### `atlas-rag chunk` - Single Document Processing
- Process individual documents (PDF, DOCX, TXT, HTML, Markdown, Images)
- Advanced OCR with multi-engine cascade (EasyOCR → PaddleOCR → pytesseract)
- Intelligent chunking strategies: semantic, sentence, token
- Configurable parameters: token limits (50-2000), overlap (0-500)
- Export formats: JSON, JSONL, CSV
- Terminal display with rich formatting (`--show`)
- Quality detection: automatically rejects unreadable documents

#### `atlas-rag batch` - Batch Processing
- Process multiple files from a directory
- Pattern matching: `*.pdf`, `*.txt`, `*.*`
- Recursive directory processing
- **Per-file output** (default): One chunk file per document
- **Single-file output** (optional): Combine all chunks
- **Automatic retry**: Up to 3 attempts with exponential backoff (1s, 2s, 4s)
- **Interactive error handling**:
  - `interactive` mode: Prompt user on each error (default)
  - `auto-continue` mode: Continue on errors (CI/CD)
  - `auto-stop` mode: Stop on first error (validation)
  - `auto-skip` mode: Skip failed files automatically
- **Complete history**: Every run saved to `~/.atlasrag/history/`
- Detailed progress reporting with Rich UI

#### `atlas-rag retry` - Retry Failed Files
- Retry failed files from previous runs
- Show run history with `--show`
- Rerun specific runs by ID
- Preserves original configuration
- Incremental processing support

#### `atlas-rag ingest` - Vector Store Integration
- Ingest chunks into Qdrant vector store
- Collection management (create, update)
- Batch ingestion support
- Configurable embedding models

#### `atlas-rag eval` - Quality Evaluation
- Evaluate chunking strategies
- Compare multiple strategies side-by-side
- Metrics: coverage, overlap, coherence
- Export evaluation results

#### `atlas-rag info` - System Information
- Display system configuration
- Check dependencies
- Show available resources

---

## Features

### 📄 Document Processing
- **Universal loader**: Supports PDF, DOCX, ODT, TXT, HTML, Markdown, JPEG, PNG
- **Advanced OCR cascade**:
  - Primary: EasyOCR (best quality, multi-language)
  - Fallback 1: PaddleOCR (fast, complex layouts)
  - Fallback 2: pytesseract (most tolerant)
- **Multi-language support**: French, English, German, Spanish, Italian, Portuguese, and more
- **Quality detection**: 90% readability threshold with word-level validation
- **Automatic fallback**: Graceful degradation between OCR engines

### ✂️ Intelligent Chunking
- **LangChain integration**: RecursiveCharacterTextSplitter for semantic chunking
- **Multiple strategies**:
  - `semantic` - Context-aware splitting (default)
  - `sentence` - Sentence-based splitting
  - `token` - Fixed token-based splitting
- **Configurable parameters**:
  - Token limits: 50-2000 tokens per chunk
  - Overlap: 0-500 tokens
  - Model selection for tokenization
- **Rich metadata**:
  - Source file path and name
  - Chunk index and total count
  - Token count and strategy used
  - Processing timestamps

### 🔄 Production-Ready Batch Processing
- **Automatic retry mechanism**:
  - Exponential backoff: 1s, 2s, 4s delays
  - Configurable retry strategies: exponential, linear, fixed
  - Fatal error detection: skip non-retriable errors
  - Detailed retry logging
- **Interactive error handling**:
  - 4 modes: interactive, auto-continue, auto-stop, auto-skip
  - User-friendly prompts with context
  - Decision tracking
- **Complete history tracking**:
  - Run IDs with microsecond precision
  - Status tracking: pending, processing, success, failed, skipped, aborted
  - Detailed metrics: duration, retries, errors per file
  - Configuration persistence for reproducibility
- **Per-file output** (default behavior):
  - One chunk file per source document
  - Clear traceability: `doc1_chunks.jsonl`, `doc2_chunks.jsonl`
  - Easier reprocessing of individual files
- **Single-file output** (optional):
  - Combine all chunks into one file
  - Useful for bulk processing

### 💾 Export & Storage
- **Multiple export formats**:
  - JSON: Pretty-printed, human-readable
  - JSONL: Streaming format for large datasets
  - CSV: Excel-compatible, tabular format
- **Vector store integration**:
  - Direct ingestion into Qdrant
  - Collection management
- **No database required**: Pure file-based export for easy sharing
- **Git-friendly**: Version control your datasets
- **ETL integration**: Easy pipeline integration

### ⚙️ Configuration System
- **Hierarchical configuration**:
  1. CLI flags (highest priority)
  2. Environment variables
  3. YAML configuration file (`~/.atlasrag/config.yml`)
  4. Default values (lowest priority)
- **Example configuration**: `config.example.yml` with detailed comments
- **Easy customization**: Override any setting via command line
- **Validation**: Automatic validation of configuration values

### 🧪 Quality & Testing
- **129 tests**: Unit, integration, and E2E tests
- **96% coverage**: Comprehensive test coverage
- **Pre-commit hooks**: Automated quality checks
  - Black (code formatting)
  - Ruff (linting)
  - Bandit (security)
  - YAML linting
- **CI/CD ready**: Auto-continue mode for unattended batch processing
- **Security audited**: Complete security best practices documentation

---

## Technical Details

### Architecture
- **Modular design**: Clear separation of concerns
- **Pipeline pattern**: Extensible processing pipeline
- **Status tracking**: Comprehensive state management
- **Error handling**: Graceful degradation with detailed logging

### Dependencies
- **Python**: 3.10, 3.11, 3.12
- **Core libraries**:
  - LangChain (text splitting, document loading)
  - EasyOCR (primary OCR engine)
  - PaddleOCR (fallback OCR engine)
  - Unstructured (universal document parsing)
  - Typer (CLI framework)
  - Rich (terminal formatting)
- **System dependencies**:
  - tesseract-ocr (OCR)
  - poppler-utils (PDF processing)

### Performance
- **Text documents**: ~100-200 docs/minute
- **PDFs with OCR**: ~5-10 docs/minute
- **OCR accuracy**: 95%+ with EasyOCR on clear scans
- **Chunk quality**: 90% readability threshold enforced
- **Batch processing**: Parallel-ready with retry mechanism

---

## Documentation

- **Quick Start Guide**: Get started in 5 minutes
- **User Guide**: Complete CLI documentation
- **Architecture**: System design and workflows
- **Data Formats**: Export formats specification
- **Security**: Security best practices and audits
- **Configuration**: Configuration reference

---

## Known Limitations

- **ODT format**: Not officially supported (will be rejected)
- **Large files**: Memory usage scales with file size
- **OCR speed**: Can be slow on low-end hardware
- **NumPy compatibility**: Requires NumPy 1.x for OCR support

---

## Future Plans

See [GitHub Issues](https://github.com/horiz-data/ragstudio/issues) for planned features and enhancements.

Potential improvements:
- Parallel batch processing
- More export formats
- Additional OCR engines
- Improved performance
- More chunking strategies
- Plugin system

---

## Versioning

This project uses [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

---

## Migration Guide

This is the first public release. No migration needed.

---

**Release Date**: January 28, 2025
**Status**: Beta
**License**: MIT
**Python**: 3.10, 3.11, 3.12

---

**Legend**:
- 🎉 Major release
- ✨ New feature
- 🐛 Bug fix
- 📝 Documentation
- 🔧 Maintenance
- ⚡ Performance improvement
- 🔒 Security fix
