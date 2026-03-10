# 1. load cross encoder model
# 2. Accept query and candidate documents
# 3. Generate relevance scores
# 4. Return reranked results

from sentence_transformers import CrossEncoder
import heapq

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def rerank(hybrid_result, query, doc_lookup, k=10):
    results = []
    candidate_pool = []

    for doc in hybrid_result:
        candidate_pool.append((query, doc_lookup[int(doc[0])]))

    scores = model.predict(candidate_pool)

    for doc, score in zip(hybrid_result, scores):
        results.append((doc[0], float(score)))

    results = heapq.nlargest(k, results, key=lambda x: x[1])

    return results
