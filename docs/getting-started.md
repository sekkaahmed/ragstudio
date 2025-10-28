# Getting Started with Atlas-RAG

Quick guide to get started with Atlas-RAG CLI.

## Installation

### From PyPI (Recommended)

```bash
pip install atlas-rag
```

### From Source

```bash
git clone git@github.com:horiz-data/atlas-rag.git
cd atlas-rag
pip install -e .
```

## Verify Installation

```bash
atlas-rag --version
# Output: Atlas-RAG CLI version 0.1.0
```

## Quick Start

### 1. Chunk Your First Document

```bash
# Create a test file
echo "This is a test document for Atlas-RAG. It will be chunked into smaller pieces for RAG applications." > test.txt

# Chunk it
atlas-rag chunk test.txt --show
```

Output:
```
✓ Successfully chunked test.txt
   Strategy:            semantic  
   Chunks created:      1         
```

### 2. Save Chunks to File

```bash
atlas-rag chunk test.txt -o chunks.json
```

### 3. Process Multiple Files

```bash
# Create test files
mkdir documents
echo "Document 1 content..." > documents/doc1.txt
echo "Document 2 content..." > documents/doc2.txt

# Process all files
atlas-rag batch documents --auto-continue
```

### 4. Ingest to Vector Store

First, start Qdrant (requires Docker):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Then ingest:

```bash
atlas-rag ingest chunks.json --collection my_docs
```

## Basic Workflow

```bash
# 1. Chunk → 2. Ingest → 3. Query
atlas-rag chunk document.pdf -o chunks.json
atlas-rag ingest chunks.json --collection docs
# Use vector store for RAG (requires API or custom integration)
```

## Common Use Cases

### Process PDF Documents

```bash
atlas-rag chunk document.pdf --strategy semantic --show
```

### Batch Process Directory

```bash
atlas-rag batch ./pdfs --pattern "*.pdf" --recursive -o ./output
```

### Custom Chunking Parameters

```bash
atlas-rag chunk large_doc.txt \
  --strategy token \
  --max-tokens 512 \
  --overlap 100 \
  -o chunks.json
```

## Next Steps

- Read the [CLI Guide](cli-guide.md) for complete command reference
- Check [Security Documentation](security/) for security features
- See [Examples](examples.md) for more use cases

## Troubleshooting

### Command not found

```bash
# Make sure atlas-rag is installed
pip show atlas-rag

# If not, reinstall
pip install --upgrade atlas-rag
```

### Import errors

```bash
# Reinstall dependencies
pip install --upgrade atlas-rag
```

### Qdrant connection errors

```bash
# Make sure Qdrant is running
docker ps | grep qdrant

# Start Qdrant if needed
docker run -p 6333:6333 qdrant/qdrant
```

## Support

- **Documentation**: [docs/](.)
- **Issues**: [GitHub Issues](https://github.com/horiz-data/atlas-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/horiz-data/atlas-rag/discussions)

---

**Version:** 0.1.0  
**Last Updated:** October 28, 2024
