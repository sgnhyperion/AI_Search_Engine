# AI Search Engine

A hybrid search engine combining BM25 lexical search, FAISS semantic search, cross-encoder reranking, and RAG-based question answering — built on the AG News dataset.

## Architecture

```
Query
  ├── BM25 (keyword)       ──┐
  └── FAISS (semantic)     ──┴──► Hybrid Score ──► Cross-Encoder Rerank ──► Results
                                  (α=0.7, β=0.3)
                                                              │
                                                              └──► Ollama (RAG /ask)
```

**Components:**
- `search.py` / `bm25.py` — BM25 inverted index search
- `vector_search.py` / `vector_index.py` — FAISS + `all-MiniLM-L6-v2` embeddings
- `hybrid_search.py` — Weighted score fusion
- `reranker.py` — `cross-encoder/ms-marco-MiniLM-L6-v2` reranking
- `rag.py` — Retrieval-Augmented Generation via Ollama (`deepseek-r1:7b`)
- `api.py` — FastAPI server

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (only required for the `/ask` RAG endpoint)

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd AI_Search_Engine
pip install -r requirements.txt
```

### 2. Download NLTK data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

### 3. Build indexes (one-time setup)

Run these in order:

```bash
# Download AG News dataset from HuggingFace (~30k docs)
python src/data_loader.py

# Tokenize, stem, and remove stopwords
python src/text_processor.py

# Build BM25 inverted index → storage/index/ & storage/metadata/
python src/index_builder.py

# Generate embeddings and build FAISS index → storage/vector_index/
python src/vector_index.py
```

> This step is only needed once. Pre-built indexes can be committed to skip it.

### 4. (Optional) Set up Ollama for RAG

```bash
ollama pull deepseek-r1:7b
ollama serve   # starts on http://localhost:11434
```

### 5. Start the API server

```bash
uvicorn src.api:app --reload
```

Server runs at `http://localhost:8000`.

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `GET /search?q=<query>&top_k=5` | Hybrid search — returns ranked results with snippets |
| `GET /ask?q=<query>` | RAG QA — returns a generated answer with sources |

**Example:**
```bash
curl "http://localhost:8000/search?q=artificial+intelligence&top_k=3"
curl "http://localhost:8000/ask?q=What+is+machine+learning?"
```

## Project Structure

```
AI_Search_Engine/
├── src/               # Source code
├── data/
│   ├── raw/           # Raw AG News dataset
│   └── processed/     # Preprocessed documents
├── storage/
│   ├── index/         # BM25 inverted index
│   ├── metadata/      # Document metadata
│   └── vector_index/  # FAISS index + doc IDs
├── requirements.txt
└── main.py
```
