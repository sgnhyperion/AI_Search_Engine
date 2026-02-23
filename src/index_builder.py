import json
import os

# Inverted index builder

def build_inverted_index(input_path, output_path, metadata_output_path):
    inverted_index = {}
    doc_lengths = {}
    doc_freq = {}
    total_docs = 0
    total_doc_length = 0
    total_unique_terms = 0
    
    with open(input_path, "r", encoding="utf-8") as f:
        document = json.load(f)

        
    for doc in document:
        tokens = doc["tokens"]
        doc_id = doc["doc_id"]
        doc_length = doc["length"]
        
        total_docs += 1
        total_doc_length += doc_length
        doc_lengths[doc_id] = doc_length
        
        # To avoid counting same term multiple times for DF
        unique_tokens = set(tokens)
        
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = {doc_id: 1}
            else:
                if doc_id in inverted_index[token]:
                    inverted_index[token][doc_id] += 1
                else:
                    inverted_index[token][doc_id] = 1
                    
        for token in unique_tokens:
            if token not in doc_freq:
                doc_freq[token] = 1
            else:
                doc_freq[token] += 1
       
    total_unique_terms = len(inverted_index)         
    avg_doc_length = total_doc_length/total_docs
    metadata = {
        "total_docs": total_docs,
        "total_unique_terms": total_unique_terms,
        "avg_doc_length": avg_doc_length,
        "doc_lengths": doc_lengths,
        "doc_freq": doc_freq
    }
    
    os.makedirs("storage/index", exist_ok=True)
    os.makedirs("storage/metadata", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, ensure_ascii=False)
        
    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
        
    print("Indexing completed.")
    print(f"Total documents: {total_docs}")
    print(f"Average document length: {avg_doc_length:.2f}")

if __name__ == "__main__":
    build_inverted_index("data/processed/processed_ag_news.json", "storage/index/ag_news_index.json", "storage/metadata/ag_news_metadata.json")