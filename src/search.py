from src.text_processor import preprocess_text
from src.bm25 import BM25
import heapq

bm25 = BM25(
    "storage/index/ag_news_index.json",
    "storage/metadata/ag_news_metadata.json"
)

def search(query, top_k=10):
    query_tokens = preprocess_text(query)
    
    scores = bm25.score(query_tokens)
    
    # ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    ranked = heapq.nlargest(top_k, scores.items(), key=lambda x:x[1])
    
    return ranked
    