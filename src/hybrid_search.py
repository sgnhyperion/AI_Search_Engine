# User query
# ↓
# 1️⃣ run BM25 search -> top 50 docs
# 2️⃣ run vector search -> top 50 docs
# 3️⃣ merge candidate docs
# 4️⃣ normalize scores -> [bm25: (bm25/max(bm25), similarity: (1/(1+distance))]
# 5️⃣ compute hybrid score -> α * normalized_bm25 + β * semantic_similarity : (α = 0.7, β = 0.3)
# 6️⃣ return ranked results

import heapq
from src.search import search
from src.vector_search import vector_search
from src.vector_index import generate_embeddings


def hybrid_search(query, k=10, alpha=0.7, beta=0.3):
    n = 10 * k

    bm25_result = search(query, n)

    search_vector = generate_embeddings([query])
    vector_result = vector_search(search_vector, n)

    bm25_dict = {doc: score for doc, score in bm25_result}
    vector_dict = {doc["doc_id"]: 1 / (1 + doc["distance"]) for doc in vector_result}

    candidates = set(bm25_dict) | set(vector_dict)

    max_bm25 = max(bm25_dict.values()) if bm25_dict else 1

    results = []

    for doc in candidates:
        bm25_score = bm25_dict.get(doc, 0) / max_bm25
        vector_score = vector_dict.get(doc, 0)

        hybrid_score = alpha * bm25_score + beta * vector_score

        results.append((doc, hybrid_score))

    results = heapq.nlargest(k, results, key=lambda x: x[1])

    return results


if __name__ == "__main__":
    results = hybrid_search("Market is going to crash on monday")
    print(results)
