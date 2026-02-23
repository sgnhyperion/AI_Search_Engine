import json
import os

# Inverted index builder

def buildInvertedIndex(input_path, output_path):
    indexed_doc = {}
    
    with open(input_path, "r", encoding="utf-8") as f:
        document = json.load(f)

        
    for doc in document:
        tokens = doc["tokens"]
        
        for token in set(tokens):
            if token not in indexed_doc:
                indexed_doc[token] = [doc["doc_id"]]
            else:
                indexed_doc[token].append(doc["doc_id"])
    
    os.makedirs("storage/index", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(indexed_doc, f, ensure_ascii=False)

if __name__ == "__main__":
    buildInvertedIndex("data/processed/processed_ag_news.json", "storage/index/ag_news_index.json")