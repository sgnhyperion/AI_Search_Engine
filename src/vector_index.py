# 1. Generate embeddings
# 2. Create vector index -> build faiss index
# 3. Save to disk -> storage/vector_index -> faiss.index, doc_ids.json

import os
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def generate_embeddings(text):
    embeddings = model.encode(text)
    return embeddings


def build_faiss_index(input_path, output_path, doc_id_listPath):
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
        
    formatted_input = []
    doc_id_list = []
    
    for doc in input_data:
        for doc_id, text in doc:
            formatted_input.append(text)
            doc_id_list.append(doc_id)
    
    embeddings = generate_embeddings(formatted_input)
    
    
    
    # os.makedirs("storage/vector_index", exist_ok=True)
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(embeddings, f, ensure_ascii=False)
        
    # with open(doc_id_listPath, "w", encoding="utf-8") as f:
    #     json.dump(doc_id_list, f, ensure_ascii=False)


if __name__ == "__main__":
    build_faiss_index("data/raw/ag_news.json", "storage/vector_index/faiss.index")