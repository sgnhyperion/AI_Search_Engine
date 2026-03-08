# 1. Generate embeddings
# 2. Create vector index -> build faiss index
# 3. Save to disk -> storage/vector_index -> faiss.index, doc_ids.json

import os
import json
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
from tqdm import tqdm
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def generate_embeddings(texts):
    
    embeddings = []
    
    for text in tqdm(texts, desc="Generating embeddings"):
        emb = model.encode(text)
        embeddings.append(emb)
        
    return embeddings


def build_faiss_index(input_path, output_path, doc_ids_path):
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
        
    formatted_input = []
    doc_ids = []
    
    for doc in input_data:
        doc_id = doc["doc_id"]
        text = doc["text"]
        
        formatted_input.append(text)
        doc_ids.append(doc_id)
    
    print("Number of documents:", len(formatted_input))
    
    # dataframe = pd.DataFrame(formatted_input, columns=['text'])
    embeddings = np.array(generate_embeddings(formatted_input)).astype("float32")
    
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    os.makedirs("storage/vector_index", exist_ok=True)
    faiss.write_index(index, output_path)
    
    with open(doc_ids_path, "w") as f:
        json.dump(doc_ids, f)
        
    print("FAISS index saved to:", output_path)
    print("Doc IDs saved to:", doc_ids_path)


if __name__ == "__main__":
    build_faiss_index("data/raw/ag_news.json", "storage/vector_index/faiss.index", "storage/vector_index/doc_ids.json")