from fastapi import FastAPI
from src.search import search
import json

app = FastAPI()

with open("data/raw/ag_news.json", "r", encoding="utf-8") as f:
    documents = json.load(f)
    
doc_lookup = {doc["doc_id"]: doc["text"] for doc in documents}

def generate_snippet(text, query, window=60):
    text_lower = text.lower()
    query_words =  query.lower().split()
    
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1:
            start = max(0, pos - window)
            end = min(len(text), pos + window)
            return text[start:end] + "..."
        
    return text[:120] + "..."
    


@app.get("/")
def home():
    return {"message": "Search Engine API is running"}

@app.get("/search")
def search_endpoint(q: str, top_k: int = 10):
    results  = search(q, top_k)
    
    return {
        "query": q,
        "results": [
            {"doc_id": doc_id, "score": score}
            for doc_id, score in results
        ]
    }