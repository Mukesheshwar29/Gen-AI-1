# StudyMate (Flask) — PDF-based Q&A with RAG

A Flask implementation of a Retrieval-Augmented Generation (RAG) system over uploaded PDFs.
It uses:
- PyMuPDF for text extraction
- SentenceTransformers (all-MiniLM-L6-v2) + FAISS for semantic retrieval
- IBM watsonx.ai (Mixtral-8x7B-Instruct) for grounded answer generation

## Quickstart

1) Create and activate a Python 3.10+ virtual environment.

```bash
pip install -r requirements.txt
```

2) Copy `.env.example` to `.env` and fill your IBM credentials:
```
IBM_API_KEY=...
IBM_PROJECT_ID=...
IBM_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=anything
```

3) Run the server:
```bash
python app.py
```
Open http://localhost:5000

## Notes
- Chunks are ~500 words with 100-word overlap (configurable in `rag_engine.py`).
- Top-3 relevant chunks are sent to the LLM with a strict, no-hallucination prompt.
- Q&A history can be downloaded as a text transcript.
