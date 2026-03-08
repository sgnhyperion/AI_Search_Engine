import faiss
import json
from vector_index import generate_embeddings
import numpy as np
import pandas as pd

index = faiss.read_index("storage/vector_index/faiss.index")
with open("storage/vector_index/doc_ids.json") as f:
    doc_ids = json.load(f)

def semantic_search(text, k=5):
    search_vector = generate_embeddings([text])
    _vector = np.array(search_vector).astype("float32")
    faiss.normalize_L2(_vector)

    distances, ann = index.search(_vector, k=k)
    
    results = []
    for dist, idx in zip(distances[0], ann[0]):
        results.append({
            "doc_id": doc_ids[idx],
            "distance": float(dist)
        })

    return results

    
if __name__ == "__main__":
    result = semantic_search("I am going becoming successful, awesome and highly skilled")
    print(result)