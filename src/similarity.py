"""Pure NumPy cosine similarity matrix calculation."""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two matrices using pure NumPy.

    Args:
        a: Matrix of shape (M, D) - e.g. query embeddings.
        b: Matrix of shape (N, D) - e.g. document chunk embeddings.

    Returns:
        Similarity matrix of shape (M, N).
    """
    # Ensure 2D arrays
    if a.ndim == 1:
        a = np.expand_dims(a, axis=0)
    if b.ndim == 1:
        b = np.expand_dims(b, axis=0)

    # Compute dot product
    dot_product = np.dot(a, b.T)

    # Compute L2 norms
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)

    # Avoid division by zero
    norm_a = np.where(norm_a == 0, 1e-10, norm_a)
    norm_b = np.where(norm_b == 0, 1e-10, norm_b)

    # Cosine similarity matrix
    similarity = dot_product / (norm_a * norm_b.T)
    return similarity
