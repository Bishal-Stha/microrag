"""Unit tests for MicroRAG components (chunking, similarity, rag_engine)."""

import numpy as np
import pytest

from src.chunking import chunk_text, chunk_documents
from src.similarity import cosine_similarity
from src.rag_engine import BareMetalRAG


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short():
    text = "Hello world"
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    assert chunks == ["Hello world"]


def test_chunk_text_sliding_window():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert len(chunks) == 3
    assert chunks[0] == "one two three four"
    assert chunks[1] == "four five six seven"
    assert chunks[2] == "seven eight nine ten"


def test_chunk_documents():
    docs = ["Doc one has some words.", "Doc two has other words."]
    chunks = chunk_documents(docs, chunk_size=10, overlap=2)
    assert len(chunks) >= 2


def test_cosine_similarity_identical():
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    sim = cosine_similarity(a, b)
    assert sim.shape == (1, 1)
    assert np.isclose(sim[0, 0], 1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    sim = cosine_similarity(a, b)
    assert np.isclose(sim[0, 0], 0.0)


def test_cosine_similarity_zero_vector():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 1.0]])
    sim = cosine_similarity(a, b)
    assert not np.isnan(sim[0, 0])


def test_rag_engine_lifecycle():
    rag = BareMetalRAG(model_name="all-MiniLM-L6-v2")
    documents = [
        "Python is a high-level programming language known for its readability.",
        "NumPy provides support for large, multi-dimensional arrays and matrices."
    ]
    rag.ingest(documents, chunk_size=10, overlap=2)
    assert len(rag.chunks) > 0
    assert rag.embeddings.shape[0] == len(rag.chunks)

    results = rag.retrieve("Tell me about Python", top_k=1)
    assert len(results) == 1
    assert "rank" in results[0]
    assert "chunk_index" in results[0]
    assert "text" in results[0]
    assert "score" in results[0]
    assert results[0]["rank"] == 1
