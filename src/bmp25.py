import math
import json

class BM25:
    def __init__(self, index_path, metadata_path, k1=1.5, b=0.75):
        with open(index_path, "r", encoding="utd-8") as f:
            self.index = json.load(f)
            
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        self.doc_lengths = metadata["doc_lengths"]
        self.doc_freq = metadata["doc_freq"]
        self.total_docs = metadata["total_docs"]
        self.avg_doc_length = metadata["avg_doc_length"]
        
        self.k1 = k1
        self.b = b
        
        
    def compute_idf(self, term):
        df = self.doc_freq.get(term, 0)
        
        if df==0:
            return 0
        
        return math.log((self.total_docs - df + 0.5)/(df + 0.5) +1)
    
    