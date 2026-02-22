# 1. load dataset
# 2. extract only text field
# 3. convert it into our own internal format
# 4. Save it as JSON inside data/raw/ag_news.json   

from datasets import load_dataset
import json, os

def load_ag_news():
    """
        Loads AG news dataset from huggingface and returns only the training split.
    """
    
    dataset = load_dataset("ag_news")
    train_data = dataset["train"]
    
    return train_data

def save_raw_data(dataset, output_path):
    
    """
        Saves dataset to JSON file in our own simplified format
    """
    
    documents = []
    
    for idx, item in enumerate(dataset):
        doc = {
            "doc_id": idx,
            "text": item["text"]
        }
        documents.append(doc)
        
    with open(output_path, "w",  encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(documents)} documents to {output_path}")
    
    
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)    
    datasets = load_ag_news()
    
    
    save_raw_data(datasets, "data/raw/ag_news.json")
    
    
    

