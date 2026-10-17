"""CLI runner with query testing and optional LLM generation via OpenRouter API."""

import os
import requests
from dotenv import load_dotenv

from src.rag_engine import BareMetalRAG


def call_openrouter_llm(prompt: str, context: str, api_key: str) -> str:
    """Generate response using OpenRouter API with retrieved context."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/bare-metal-rag",
        "X-Title": "Bare-Metal RAG",
    }

    system_prompt = (
        "You are a precise technical assistant. Answer the user's question accurately "
        "using ONLY the provided context. If the answer cannot be found in the context, state "
        "that you don't know."
    )

    user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}"

    payload = {
        "model": "openai/gpt-4o-mini",  # or claude-3.5-sonnet via openrouter
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"[Error from OpenRouter API: {response.status_code} - {response.text}]"
    except Exception as e:
        return f"[Exception calling OpenRouter API: {str(e)}]"


def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    print("=== Bare-Metal RAG from Scratch ===")

    # 1. Load corpus
    corpus_path = os.path.join("data", "sample_corpus.txt")
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return

    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    # Split into paragraphs as raw documents
    documents = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]

    # 2. Initialize RAG engine
    rag = BareMetalRAG(model_name="all-MiniLM-L6-v2")

    # 3. Ingest documents
    rag.ingest(documents, chunk_size=60, overlap=15)

    # 4. Test queries
    test_queries = [
        "How does RouteOps use ONNX quantization?",
        "What technologies power Midpoint Meetup Finder?",
        "How is inter-service communication secured in RouteOps?"
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        results = rag.retrieve(query, top_k=2)

        context_parts = []
        for res in results:
            print(f"\n[Rank {res['rank']}] Chunk Index: {res['chunk_index']} | Score: {res['score']:.4f}")
            print(f"Text: {res['text']}")
            context_parts.append(res['text'])

        context = "\n\n".join(context_parts)

        # 5. LLM Generation
        if api_key and api_key != "your_openrouter_api_key_here":
            print("\nGenerating LLM response via OpenRouter...")
            answer = call_openrouter_llm(query, context, api_key)
            print(f"Answer:\n{answer}")
        else:
            print("\n[INFO] OPENROUTER_API_KEY not configured. Skipping LLM generation call.")
            print(f"Retrieved Context for Generation:\n{context}")

    print("\n" + "=" * 60)
    print("Verification completed successfully!")
    print(f"Embedding matrix shape: {rag.embeddings.shape}")


if __name__ == "__main__":
    main()
