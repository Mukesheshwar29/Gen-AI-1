# llm_ollama.py
import requests
import os
from typing import List, Dict

class OllamaLLM:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")

    @staticmethod
    def build_prompt(question: str, contexts: List[Dict]) -> str:
        context_block = "\n\n".join([f"- Source: {c['source']}\n{c['text']}" for c in contexts])
        return f"""You are StudyMate, a helpful academic assistant. Answer strictly using the context.
If the answer is not in the context, say you don't know and suggest where in the PDFs to look.

Context:
{context_block}

Question: {question}

Answer concisely (4-8 lines) and include key terms. Avoid speculation."""

    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:
            return f"[LLM error] {e}"

