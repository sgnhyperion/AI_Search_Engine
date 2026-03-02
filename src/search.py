from text_processor import preprocess_text
from bm25 import BM25

bm25 = BM25(
    "storage/index/ag_news_index.json",
    "storage/metadata/ag_news_metadata.json"
)

def search(query, top_k=10):
    query_tokens = preprocess_text(query)
    
    scores = bm25.score(query_tokens)
    
    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    
    return ranked[:top_k]

if __name__ == "__main__":
    while True:
        query = input("Enter your search query: ")
        results = search(query)
        
        for doc_id, score in results:
            print(f"Doc ID: {doc_id}, Score: {score:.4f}, sent: ")
    