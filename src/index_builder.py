import json
import os

# Inverted index builder

def buildInvertedIndex(input_path, output_path):
    indexed_doc = {}
    
    with open(input_path, "r", encoding="utf-8") as f:
        document = json.load(f)

        
    for doc in document:
        tokens = doc["tokens"]
        doc_id = doc["doc_id"]
        
        for token in tokens:
            if token not in indexed_doc:
                indexed_doc[token] = {doc_id: 1}
            else:
                if doc_id in indexed_doc[token]:
                    indexed_doc[token][doc_id] += 1
                else:
                    indexed_doc[token][doc_id] = 1
    
    os.makedirs("storage/index", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(indexed_doc, f, ensure_ascii=False)

if __name__ == "__main__":
    buildInvertedIndex("data/processed/processed_ag_news.json", "storage/index/ag_news_index.json")