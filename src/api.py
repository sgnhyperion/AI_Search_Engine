from fastapi import FastAPI
from src.search import search

app = FastAPI()


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