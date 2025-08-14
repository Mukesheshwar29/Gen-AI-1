from __future__ import annotations
import os, re
import numpy as np
import fitz  # PyMuPDF
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

class RAGEngine:
    def __init__(self, chunk_words: int = 500, overlap: int = 100, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []  # list of dicts: {'text','source','chunk_id','embedding'}
        self.embeddings = None  # np.ndarray

    # --------- ingestion ---------
    def ingest_pdfs(self, paths):
        new_chunks = []
        for path in paths:
            text = self._extract_text(path)
            chunks = self._chunk_text(text, source=os.path.basename(path))
            new_chunks.extend(chunks)

        if not new_chunks:
            return

        # Embed new chunks
        new_embeddings = self.model.encode([c["text"] for c in new_chunks], convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

        # Append
        start_id = len(self.chunks)
        for i, c in enumerate(new_chunks):
            c["chunk_id"] = start_id + i
            c["embedding"] = new_embeddings[i]
        self.chunks.extend(new_chunks)

        # Rebuild FAISS
        all_emb = np.array([c["embedding"] for c in self.chunks], dtype="float32")
        self.embeddings = all_emb
        dim = all_emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine since embeddings normalized
        self.index.add(all_emb)

    def _extract_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        pages = []
        for p in doc:
            pages.append(p.get_text("text"))
        text = "\n".join(pages)
        # Simple cleanup
        text = re.sub(r"\s+", " ", text)
        return text

    def _chunk_text(self, text: str, source: str):
        words = text.split()
        chunks = []
        step = max(1, self.chunk_words - self.overlap)
        for start in range(0, len(words), step):
            end = start + self.chunk_words
            piece = " ".join(words[start:end]).strip()
            if piece:
                chunks.append({"text": piece, "source": source})
        return chunks

    # --------- retrieval ---------
    def retrieve(self, query: str, k: int = 3):
        if self.index is None or self.embeddings is None or len(self.chunks) == 0:
            return []
        q_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append({"text": c["text"], "source": c["source"], "score": float(score)})
        return results

    def stats(self):
        return {
            "documents": len(set(c["source"] for c in self.chunks)),
            "chunks": len(self.chunks),
            "embedding_dim": int(self.embeddings.shape[1]) if self.embeddings is not None else 0,
        }

    def corpus_size(self):
        return len(self.chunks)
