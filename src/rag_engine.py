"""BareMetalRAG engine class handling ingestion, encoding, and retrieval."""

import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

from src.chunking import chunk_documents
from src.similarity import cosine_similarity


class BareMetalRAG:
    """A bare-metal RAG engine built with SentenceTransformers and pure NumPy."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG engine with a lightweight embedding model."""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])

    def ingest(self, documents: List[str], chunk_size: int = 80, overlap: int = 20) -> None:
        """Ingest raw documents, split into chunks, and compute embeddings.

        Args:
            documents: List of raw document strings.
            chunk_size: Word count per chunk.
            overlap: Overlapping words between chunks.
        """
        print(f"Ingesting {len(documents)} documents...")
        self.chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        print(f"Generated {len(self.chunks)} text chunks.")

        if self.chunks:
            print("Encoding chunks into vector embeddings...")
            self.embeddings = self.model.encode(self.chunks, show_progress_bar=False)
            print(f"Embedding matrix shape: {self.embeddings.shape}")
        else:
            self.embeddings = np.array([])

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for a given query using pure NumPy cosine similarity.

        Args:
            query: User search query string.
            top_k: Number of top matching chunks to return.

        Returns:
            List of dictionaries containing rank, chunk text, and similarity score.
        """
        if not self.chunks or self.embeddings.size == 0:
            return []

        # Encode query
        query_embedding = self.model.encode([query], show_progress_bar=False)

        # Compute cosine similarity matrix via pure NumPy
        sim_matrix = cosine_similarity(query_embedding, self.embeddings)  # shape (1, N)
        scores = sim_matrix[0]  # shape (N,)

        # Get top-k indices sorted by descending score
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "chunk_index": int(idx),
                    "text": self.chunks[idx],
                    "score": float(scores[idx]),
                }
            )

        return results
