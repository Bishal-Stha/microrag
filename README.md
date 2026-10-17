# MicroRAG 🚀

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Pure%20NumPy%20Vector%20Math-orange.svg)](https://numpy.org/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-green.svg)](https://www.sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MicroRAG** is a clean, zero-dependency-bloat Retrieval-Augmented Generation (RAG) system built entirely from scratch using pure Python, NumPy, and SentenceTransformers. 

Designed without heavyweight frameworks like LangChain or LlamaIndex, MicroRAG provides a transparent, high-performance, and understandable blueprint for document chunking, vector embedding generation, custom vector similarity search via pure NumPy matrix operations, and LLM synthesis.

---

## 📂 Project Structure

```text
microrag/
├── src/
│   ├── __init__.py
│   ├── chunking.py      # Sliding window text chunking implementation
│   ├── similarity.py    # Pure NumPy cosine similarity matrix calculation
│   ├── rag_engine.py    # BareMetalRAG core engine (ingest, encode, retrieve)
│   └── main.py          # CLI runner & OpenRouter LLM generation integration
├── tests/
│   └── test_rag.py      # Comprehensive unit test suite (pytest)
├── data/
│   └── sample_corpus.txt # Sample technical corpus for testing
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

---

## 🧠 Core Components

### 1. Sliding Window Chunking (`src/chunking.py`)
Splits large text documents into overlapping chunks based on word count and configurable window overlap to preserve semantic context across chunk boundaries.

### 2. Pure NumPy Cosine Similarity (`src/similarity.py`)
Computes the cosine similarity matrix between query embeddings `(M, D)` and document chunk embeddings `(N, D)` in pure NumPy:
$$\text{Similarity} = \frac{A \cdot B^T}{\|A\|_2 \|B\|_2}$$
Includes robust division-by-zero protection and handles 1D/2D array shapes dynamically.

### 3. Bare-Metal RAG Engine (`src/rag_engine.py`)
The `BareMetalRAG` class orchestrates:
- Loading lightweight SentenceTransformers models (e.g., `all-MiniLM-L6-v2`).
- Ingesting raw document corpuses and generating dense vector embeddings.
- Retrieving the top-$k$ most relevant context chunks ranked by similarity score.

### 4. CLI Runner & LLM Generation (`src/main.py`)
Executes end-to-end RAG workflows over sample technical documents and optionally queries advanced LLMs (via OpenRouter API, e.g., `openai/gpt-4o-mini`) using the retrieved context.

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Bishal-Stha/microrag.git
   cd microrag
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables (Optional for LLM Generation)**:
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

---

## 🚀 Usage

Run the RAG verification pipeline and test queries:
```bash
python -m src.main
```

---

## 🧪 Running Unit Tests

MicroRAG comes with a comprehensive test suite covering chunking edge cases, vector similarity mathematics, and end-to-end RAG engine lifecycles.

Run tests using `pytest`:
```bash
PYTHONPATH=. pytest
```

---

## 🤝 Author

**Bishal Shrestha**  
GitHub: [@Bishal-Stha](https://github.com/Bishal-Stha)

---

## 📄 License

This project is licensed under the MIT License.
