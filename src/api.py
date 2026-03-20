from fastapi import FastAPI
from src.hybrid_search import hybrid_search
from src.text_processor import preprocess_text
import json
from collections import defaultdict

app = FastAPI()

with open("data/raw/ag_news.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

doc_lookup = {doc["doc_id"]: doc["text"] for doc in documents}


from collections import Counter

def generate_snippet(text, query, window=60):
    words = text.lower().split()
    query_words = preprocess_text(query)

    if not words:
        return ""

    query_set = set(query_words)

    left = 0
    curr_score = 0
    max_score = 0
    best_start = 0

    # initialize first window
    for right in range(min(window, len(words))):
        if words[right] in query_set:
            curr_score += 1

    max_score = curr_score
    best_start = 0

    # slide window
    for right in range(window, len(words)):
        
        # remove left word
        if words[left] in query_set:
            curr_score -= 1
        left += 1

        # add right word
        if words[right] in query_set:
            curr_score += 1

        # update best window
        if curr_score > max_score:
            max_score = curr_score
            best_start = left

    # extract snippet
    best_end = best_start + window
    snippet = " ".join(words[best_start:best_end])

    return snippet + "..."


@app.get("/")
def home():
    return {"message": "Search Engine API is running"}


@app.get("/search")
def search_endpoint(q: str, top_k: int = 10):
    results = hybrid_search(q, doc_lookup, top_k)
    formatted_results = []

    for doc_id, score in results:
        text = doc_lookup[int(doc_id)]
        snippet = generate_snippet(text, q)

        formatted_results.append(
            {"doc_id": doc_id, "score": score, "snippet": snippet, "text": text}
        )

    return {"query": q, "results": formatted_results}
