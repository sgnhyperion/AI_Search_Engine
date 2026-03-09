# User query
# ↓
# 1️⃣ run BM25 search -> top 50 docs
# 2️⃣ run vector search -> top 50 docs
# 3️⃣ merge candidate docs
# 4️⃣ normalize scores -> [bm25: (bm25/max(bm25), similarity: (1/(1+distance))]
# 5️⃣ compute hybrid score -> α * normalized_bm25 + β * semantic_similarity : (α = 0.7, β = 0.3)
# 6️⃣ return ranked results

from src.search import search
from src.vector_search import vector_search

def hybrid_search(query, k=10):
    results = []
    n = 5*k
    
    candidate_pool = set()
    bm25_result = search(query)
    vector_result = vector_search(query)
    
    for docA, docB in zip(bm25_result, vector_result):
        candidate_pool.add(docA[0])
        candidate_pool.add(docB["doc_id"])
        
        
        
    
    return results

if __name__ == "__main__":
    results = hybrid_search("Market is going to crash on monday") 
    print(results)

