from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.chunk.chunker import chunk_document
from src.workflows.io.schema import Document


def _stub_tiktoken(monkeypatch):
    class StubEncoding:
        def encode(self, text: str):
            return [ord(ch) for ch in text]

        def decode(self, tokens):
            return "".join(chr(tok) for tok in tokens)

    stub = SimpleNamespace(
        encoding_for_model=lambda model: StubEncoding(),
        get_encoding=lambda name: StubEncoding(),
    )
    monkeypatch.setattr("src.core.chunk.chunker.tiktoken", stub)


def test_chunk_document_returns_empty_for_blank_text():
    doc = Document(source_path=Path("empty.txt"), text="")
    assert chunk_document(doc) == []


def test_chunk_document_uses_fallback_when_chonkie_missing(monkeypatch):
    doc = Document(
        source_path=Path("sample.txt"),
        text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5,
    )
    monkeypatch.setattr("src.core.chunk.chunker.ChonkieChunker", None)
    _stub_tiktoken(monkeypatch)
    chunks = chunk_document(doc, max_tokens=20, overlap=5)
    assert len(chunks) >= 3
    assert all(chunk.text for chunk in chunks)


@pytest.mark.parametrize("overlap", [0, 10])
def test_chunk_document_respects_overlap(monkeypatch, overlap: int):
    text = " ".join(str(i) for i in range(120))
    doc = Document(source_path=Path("numbers.txt"), text=text)
    monkeypatch.setattr("src.core.chunk.chunker.ChonkieChunker", None)
    _stub_tiktoken(monkeypatch)
    chunks = chunk_document(doc, max_tokens=30, overlap=overlap)
    assert chunks
    if overlap == 0:
        assert len(chunks[0].text) <= len(chunks[1].text) + 20
