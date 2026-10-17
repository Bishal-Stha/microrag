"""Sliding window text chunking implementation for bare-metal RAG."""

from typing import List


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping chunks based on word count.

    Args:
        text: Input text corpus string.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = 1

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        if i + chunk_size >= len(words):
            break

    return chunks


def chunk_documents(documents: List[str], chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Process a list of documents/paragraphs into chunks."""
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)
    return all_chunks
