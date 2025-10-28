# 🚀 Atlas-RAG

**Production-ready document processing CLI for RAG applications**

Process documents, extract text with advanced OCR, chunk intelligently, and prepare data for RAG systems - all from the command line.

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/horiz-data/atlas-rag)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/horiz-data/atlas-rag)
[![Tests](https://img.shields.io/badge/tests-129%20passed-success.svg)](https://github.com/horiz-data/atlas-rag)
[![Coverage](https://img.shields.io/badge/coverage-96%25-success.svg)](https://github.com/horiz-data/atlas-rag)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 What is Atlas-RAG?

Atlas-RAG is a **command-line tool** for processing documents into chunks ready for Retrieval-Augmented Generation (RAG) systems. It handles the dirty work of document ingestion, OCR, and intelligent chunking so you can focus on building your RAG application.

**Key capabilities:**
- 📄 Universal document loading (PDF, DOCX, images, HTML, Markdown, etc.)
- 🔍 Advanced OCR with automatic fallback (EasyOCR → PaddleOCR → pytesseract)
- ✂️ Intelligent semantic chunking using LangChain
- 📦 Production-ready batch processing with auto-retry
- 💾 Multiple export formats (JSON, JSONL, CSV)
- 🗄️ Direct ingestion into Qdrant vector store

---

## ✨ Features

### 📄 Universal Document Processing
- **Supported formats**: PDF, DOCX, ODT, TXT, HTML, Markdown, Images (JPEG, PNG)
- **Smart OCR cascade**:
  1. EasyOCR (best quality, multi-language)
  2. PaddleOCR (fast, good for complex layouts)
  3. pytesseract (fallback, most tolerant)
- **Quality detection**: Automatically rejects unreadable documents
- **Multi-language**: French, English, German, Spanish, Italian, Portuguese, and more

### ✂️ Intelligent Chunking
- **Semantic chunking**: Context-aware text splitting using LangChain RecursiveCharacterTextSplitter
- **Multiple strategies**:
  - `semantic` - Smart splitting by meaning (default)
  - `sentence` - Split by sentences
  - `token` - Fixed token-based splitting
- **Configurable**: Token limits (50-2000), overlap (0-500), model selection
- **Rich metadata**: Source file, chunk index, token count, strategy, timestamps

### 🔄 Production-Ready Batch Processing
- **Automatic retry**: Up to 3 attempts with exponential backoff (1s, 2s, 4s...)
- **Interactive error handling**:
  - `interactive` - Prompt user on each error (default)
  - `auto-continue` - Continue on errors (CI/CD mode)
  - `auto-stop` - Stop on first error (validation mode)
  - `auto-skip` - Skip failed files automatically
- **Complete history**: Every run saved to `~/.atlasrag/history/`
- **Retry capability**: `atlas-rag retry` to rerun failed files only
- **Per-file output**: One chunk file per document for better traceability

### 💾 Flexible Export & Storage
- **Export formats**: JSON, JSONL (streaming), CSV (Excel-compatible)
- **Vector store integration**: Direct ingestion into Qdrant
- **No database required**: Pure file-based export for easy sharing

### ⚙️ Configuration System
- **Hierarchical config**: CLI flags > Environment variables > YAML file > Defaults
- **Example config**: `config.example.yml` with detailed documentation
- **Easy customization**: Override any setting via command line

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone git@github.com:horiz-data/atlas-rag.git
cd atlas-rag

# Install with pip
pip install -e .

# Verify installation
atlas-rag --version
```

### Basic Usage

```bash
# Process a single document
atlas-rag chunk document.pdf --show

# Process with advanced OCR for scanned documents
atlas-rag chunk scanned.pdf --advanced-ocr -o chunks.json

# Batch process a folder
atlas-rag batch ./documents --output ./chunks/

# Batch with auto-retry for CI/CD
atlas-rag batch ./documents --output ./chunks/ --auto-continue
```

---

## 💡 Usage Examples

### Single Document Processing

```bash
# Simple text file
atlas-rag chunk document.txt --show

# PDF with semantic chunking (default)
atlas-rag chunk report.pdf -o report_chunks.json

# Scanned image with OCR
atlas-rag chunk contract.jpeg --advanced-ocr --show

# Custom chunking parameters
atlas-rag chunk document.pdf \
  --strategy semantic \
  --max-tokens 500 \
  --overlap 100 \
  -o output.jsonl
```

### Batch Processing

```bash
# Process all files in a directory
atlas-rag batch ./documents --output ./chunks/

# Process only PDFs recursively
atlas-rag batch ./documents \
  --pattern "*.pdf" \
  --recursive \
  --output ./chunks/

# CI/CD mode - continue on errors
atlas-rag batch ./documents \
  --output ./chunks/ \
  --auto-continue \
  --save-history

# Per-file output (default):
# chunks/
# ├── doc1_chunks.jsonl  (25 chunks)
# ├── doc2_chunks.jsonl  (42 chunks)
# └── doc3_chunks.jsonl  (18 chunks)

# Single-file output (all chunks combined):
atlas-rag batch ./documents \
  --output ./all_chunks.jsonl \
  --single-file
```

### Retry Failed Files

```bash
# Show last failed run
atlas-rag retry --show

# Retry all failed files from last run
atlas-rag retry

# Retry specific run by ID
atlas-rag retry run_20251028_133403
```

### Vector Store Integration

```bash
# Ingest chunks into Qdrant
atlas-rag ingest chunks.jsonl \
  --collection my-docs \
  --url http://localhost:6333

# Get system info
atlas-rag info
```

### Evaluate Chunking Quality

```bash
# Evaluate chunking strategy
atlas-rag eval document.pdf \
  --strategies semantic sentence token \
  --metrics coverage overlap coherence

# Compare strategies with visualization
atlas-rag eval document.pdf --compare --output eval_results.json
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Getting Started](docs/getting-started.md)** | Installation and first steps |
| **[CLI Guide](docs/cli-guide.md)** | Complete command reference |
| **[Security](docs/security/)** | Security features and best practices |
| **[Full Documentation](docs/)** | Complete documentation index |

---

## ⚙️ Configuration

Create `~/.atlasrag/config.yml` or use CLI flags:

```yaml
# OCR settings
ocr:
  use_advanced_ocr: false
  enable_fallback: true

# Chunking settings
chunking:
  strategy: semantic
  max_tokens: 400
  overlap: 50

# Output settings
output:
  format: jsonl
  include_metadata: true
  pretty_print: true
```

**Configuration hierarchy**: CLI flags > Environment variables > YAML config > Defaults

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run CLI tests
make test-cli

# Quick validation
atlas-rag --version
atlas-rag chunk tests/data/sample.txt --show
```

**Test Coverage**: 129 tests, 96% coverage

---

## 📊 Performance

### Processing Speed
- **Text documents**: ~100-200 docs/minute
- **PDFs with OCR**: ~5-10 docs/minute (depends on page count)
- **Batch processing**: Parallel-ready with retry mechanism

### Quality Metrics
- **OCR accuracy**: 95%+ with EasyOCR on clear scans
- **Chunk quality**: 90% readability threshold enforced
- **Semantic coherence**: LangChain's RecursiveCharacterTextSplitter optimized for context

---

## 🛠️ CLI Commands

| Command | Description |
|---------|-------------|
| `atlas-rag chunk` | Process a single document |
| `atlas-rag batch` | Batch process multiple files |
| `atlas-rag retry` | Retry failed files from history |
| `atlas-rag ingest` | Ingest chunks into Qdrant |
| `atlas-rag eval` | Evaluate chunking quality |
| `atlas-rag info` | System information |

Run `atlas-rag COMMAND --help` for detailed options.

---

## 🐛 Troubleshooting

### Common Issues

**NumPy incompatibility**
```bash
# For OCR support, use NumPy 1.x
pip install "numpy<2.0"
```

**Missing system dependencies**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

**"Document unreadable" errors**
- Try lowering quality threshold: `--ocr-threshold 0.2`
- Use advanced OCR: `--advanced-ocr`
- Check document is not corrupted

**Import errors**
```bash
# Reinstall dependencies
pip install -e .
```

More help: [Getting Started Guide](docs/getting-started.md#troubleshooting)

---

## 🔧 Development

```bash
# Install dev dependencies
make install-dev

# Format code
make format

# Run linters
make lint

# Install pre-commit hooks
make pre-commit-install

# Run all CI checks
make ci-all
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines (coming soon).

---

## 📧 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/horiz-data/atlas-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/horiz-data/atlas-rag/discussions)

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - Text splitting and document loading
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Alternative OCR engine
- [Unstructured](https://github.com/Unstructured-IO/unstructured) - Document parsing
- [Typer](https://github.com/tiangolo/typer) - CLI framework
- [Rich](https://github.com/Textualize/rich) - Terminal formatting

---

**Version**: 0.1.0 | **Status**: Beta | **License**: MIT
