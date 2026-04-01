import requests
from src.hybrid_search import hybrid_search
from src.reranker import rerank

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:7b"


def build_context(docs, doc_lookup, max_chars=400):
    context = ""

    for i, (doc_id, _) in enumerate(docs, 1):
        text = doc_lookup[int(doc_id)]
        text = text[:max_chars]  # truncate for safety

        context += f"[Document {i}]\n{text}\n\n"

    return context


def generate_answer(query, doc_lookup):
    # Step 1: Hybrid retrieval
    hybrid_results = hybrid_search(query, doc_lookup, k=10)

    # Step 2: Cross-encoder reranking
    reranked = rerank(hybrid_results, query, doc_lookup, k=3)

    # Step 3: Build context
    context = build_context(reranked, doc_lookup)

    # Step 4: Prompt
    prompt = f"""
You are a strict question-answering system.

You MUST answer ONLY using the provided context.

Rules:
- Do NOT use your own knowledge
- Extract the answer from the context
- If the context does not contain the answer, say: "I don't know"
- Be specific to the context, not general definitions
- Keep answer under 3 sentences

Context:
{context}

Question:
{query}

Answer (based ONLY on context):
"""

    # Step 5: Call Ollama (DeepSeek)
    response = requests.post(
        OLLAMA_URL, json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
    )

    result = response.json()

    answer = result.get("response", "").strip()

    return {"answer": answer, "sources": [doc_id for doc_id, _ in reranked]}


# Test
if __name__ == "__main__":
    import json

    with open("data/raw/ag_news.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    doc_lookup = {doc["doc_id"]: doc["text"] for doc in documents}

    query = "What caused the financial crisis?"
    result = generate_answer(query, doc_lookup)

    print("\nAnswer:\n", result["answer"])
    print("\nSources:", result["sources"])
