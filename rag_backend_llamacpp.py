"""
RAG Backend for BatteryLab Copilot (Hugging Face + llama-cpp-python)
Uses a GGUF model from Hugging Face with llama-cpp-python for local inference.
Install: pip install llama-cpp-python
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple

# Configuration
VECTOR_DB_DIR = "rag_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RETRIEVAL_CHUNKS = 3
TEMPERATURE = 0.1
MAX_TOKENS = 300
CONTEXT_WINDOW = 2048  # Context window size for llama-cpp-python (default is 512, too small)

# Paper metadata mapping (filename stem -> title and author)
PAPER_METADATA = {
    "Severson_NatureEnergy_2019": {
        "title": "Data-driven prediction of battery cycle life before capacity degradation",
        "author": "Severson et al.",
        "year": "2019",
        "journal": "Nature Energy"
    },
    "Early cycles": {
        "title": "Early cycles",
        "author": "Unknown",
        "year": "Unknown"
    },
    "Arrhneius derived emp on Severson": {
        "title": "Arrhenius-derived empirical model",
        "author": "Unknown",
        "year": "Unknown"
    },
    "ICA for SOH at Diff Temp": {
        "title": "ICA for State of Health at Different Temperatures",
        "author": "Unknown",
        "year": "Unknown"
    },
    "SEI model": {
        "title": "SEI Model",
        "author": "Unknown",
        "year": "Unknown"
    },
    "SoH with C rate Emp": {
        "title": "State of Health with C-rate Empirical Model",
        "author": "Unknown",
        "year": "Unknown"
    },
    "wang et al": {
        "title": "Wang et al.",
        "author": "Wang et al.",
        "year": "Unknown"
    }
}


def get_paper_citation(paper_name: str) -> str:
    """Get formatted citation for a paper."""
    metadata = PAPER_METADATA.get(paper_name, {
        "title": paper_name,
        "author": "Unknown",
        "year": "Unknown"
    })
    if metadata.get("journal"):
        return f"{metadata['author']} ({metadata['year']}), \"{metadata['title']}\", {metadata['journal']}"
    else:
        return f"{metadata['author']} ({metadata['year']}), \"{metadata['title']}\""

# Hugging Face GGUF model (env or .streamlit/secrets.toml)
# Default: Llama 3.2 3B Instruct (text-only, ~2 GB) - smallest recommended option
# Alternative larger models:
# - Llama 3.2 11B Vision: ~7.8 GB (Q4_K_M) or ~21.5 GB (F16) - supports images
# - Llama 3.1 8B Instruct: ~4-5 GB (Q4_K_M) - text-only, better quality
HF_LLAMA_REPO_ID = os.getenv("HF_LLAMA_REPO_ID", "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF")
HF_LLAMA_FILENAME = os.getenv("HF_LLAMA_FILENAME", "llama-3.2-3b-instruct-q4_k_m.gguf")


def _get_hf_config() -> Tuple[str, str]:
    """Resolve repo_id and filename from env or Streamlit secrets."""
    repo_id = os.getenv("HF_LLAMA_REPO_ID", HF_LLAMA_REPO_ID)
    filename = os.getenv("HF_LLAMA_FILENAME", HF_LLAMA_FILENAME)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            repo_id = st.secrets.get("HF_LLAMA_REPO_ID", repo_id)
            filename = st.secrets.get("HF_LLAMA_FILENAME", filename)
    except Exception:
        pass
    return repo_id, filename


class BatteryLabRAG:
    """RAG system for BatteryLab research papers using llama-cpp-python + Hugging Face GGUF."""

    def __init__(self, repo_id: Optional[str] = None, filename: Optional[str] = None):
        cfg_repo, cfg_filename = _get_hf_config()
        self.repo_id = repo_id if repo_id is not None else cfg_repo
        self.filename = filename if filename is not None else cfg_filename

        self._load_embedding_model()
        self._load_llm()
        self._load_vector_db()

    def _load_embedding_model(self):
        import time
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if attempt == 0:
                    try:
                        self.embedding_model = SentenceTransformer(
                            EMBEDDING_MODEL, device="cpu", local_files_only=True
                        )
                        return
                    except Exception:
                        pass
                self.embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL, device="cpu", local_files_only=False
                )
                return
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to load embedding model: {str(e)}")
                time.sleep(1)

    def _load_llm(self):
        """Load GGUF model from Hugging Face.
        
        The GGUF file is cached locally after first download:
        - Windows: C:\\Users\\<username>\\.cache\\huggingface\\hub\\
        - Linux/Mac: ~/.cache/huggingface/hub/
        
        Subsequent runs use the cached file (no re-download).
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Install llama-cpp-python: pip install llama-cpp-python\n"
                "See: https://github.com/abetlen/llama-cpp-python"
            )
        try:
            self.llm = Llama.from_pretrained(
                repo_id=self.repo_id,
                filename=self.filename,
                n_ctx=CONTEXT_WINDOW,  # Set context window size
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Hugging Face model {self.repo_id} / {self.filename}: {str(e)}\n"
                "Ensure the repo and filename exist on Hugging Face (GGUF).\n"
                f"Note: First run downloads to cache (Windows: %USERPROFILE%\\.cache\\huggingface\\hub\\)"
            ) from e

    def _load_vector_db(self):
        chunks_path = os.path.join(VECTOR_DB_DIR, "chunks.json")
        index_path = os.path.join(VECTOR_DB_DIR, "faiss.index")
        if not os.path.exists(chunks_path) or not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Vector database not found. Run rag_ingestion.py first.\n"
                f"Expected: {chunks_path}, {index_path}"
            )
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.index = faiss.read_index(index_path)

    def _retrieve_chunks(self, query: str, k: int = MAX_RETRIEVAL_CHUNKS) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        query_embedding = query_embedding.astype("float32")
        search_k = min(k * 3, self.index.ntotal)
        _, indices = self.index.search(query_embedding, search_k)
        selected_indices = []
        for idx in indices[0]:
            if len(selected_indices) >= k:
                break
            chunk = self.chunks[idx]
            paper = chunk["paper"]
            paper_count = sum(1 for i in selected_indices if self.chunks[i]["paper"] == paper)
            if paper_count < 2 or len(selected_indices) < k:
                selected_indices.append(int(idx))
        return [self.chunks[i] for i in selected_indices[:k]]

    def _generate_answer(self, query: str, context_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        if not context_chunks:
            return "Not covered in my knowledge core.", []

        context_parts = []
        sources = []
        paper_citations = {}  # Map paper names to citations
        
        for chunk in context_chunks:
            paper_name = chunk["paper"]
            context_parts.append(chunk["text"])
            sources.append({"paper": paper_name, "section": chunk["section"]})
            # Build citation mapping
            if paper_name not in paper_citations:
                paper_citations[paper_name] = get_paper_citation(paper_name)

        # Truncate context to fit within context window
        # Estimate: ~4 chars per token, reserve ~300 tokens for prompts/response
        max_context_tokens = CONTEXT_WINDOW - 300
        max_context_chars = max_context_tokens * 3  # Conservative: 3 chars per token
        truncated_parts = []
        chars_used = 0
        for i, (text, chunk) in enumerate(zip(context_parts, context_chunks)):
            paper_name = chunk["paper"]
            citation = paper_citations[paper_name]
            
            # Limit each chunk to ~800 chars (~200 tokens) to fit multiple chunks
            chunk_max = 800
            if len(text) > chunk_max:
                text = text[:chunk_max] + "..."
            
            # Format with paper citation instead of "Source X"
            citation_header = f"[{citation}]\n"
            
            if chars_used + len(text) + len(citation_header) > max_context_chars:
                # Try to fit at least part of this chunk
                remaining = max_context_chars - chars_used - len(citation_header) - 50
                if remaining > 200:  # Only add if meaningful chunk remains
                    text = text[:remaining] + "..."
                    truncated_parts.append(f"{citation_header}{text}")
                break
            truncated_parts.append(f"{citation_header}{text}")
            chars_used += len(text) + len(citation_header) + 10  # +10 for spacing overhead
        
        context = "\n\n".join(truncated_parts)

        system_prompt = (
            "Answer questions about battery research using ONLY the provided context. "
            "Be concise and factual. Cite sources by paper author and year (e.g., 'Severson et al. (2019)'). "
            "Use the exact citation format shown in brackets before each context section. "
            "If information isn't in the context, say 'Not covered in my knowledge core.'"
        )
        user_prompt = f"Context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            out = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            # OpenAI-like shape: choices[0].message.content
            if isinstance(out, dict) and "choices" in out and out["choices"]:
                msg = out["choices"][0].get("message", {})
                answer = (msg.get("content") or "").strip()
            else:
                answer = str(out).strip()
            if not answer:
                answer = "No answer generated."

            seen = set()
            unique_sources = []
            for s in sources:
                key = (s["paper"], s["section"])
                if key not in seen:
                    unique_sources.append(s)
                    seen.add(key)
            return answer, unique_sources

        except Exception as e:
            err = str(e).lower()
            if "timeout" in err or "timed out" in err:
                return (
                    "⚠️ **Request timeout** — try a shorter question or a smaller GGUF model.",
                    [],
                )
            if "context window" in err or "exceed" in err:
                return (
                    f"⚠️ **Context window exceeded** — the question/context is too long. "
                    f"Try a shorter, more specific question.",
                    [],
                )
            return f"Error generating answer: {e}", []

    def query(self, message: str) -> Dict:
        chunks = self._retrieve_chunks(message, k=MAX_RETRIEVAL_CHUNKS)
        answer, sources = self._generate_answer(message, chunks)
        return {"answer": answer, "sources": sources}


_rag_instance: Optional[BatteryLabRAG] = None


def get_rag_instance() -> BatteryLabRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = BatteryLabRAG()
    return _rag_instance


def chat_api(message: str) -> Dict:
    if not message or not message.strip():
        return {"answer": "Please provide a question.", "sources": []}
    if len(message) > 1000:
        return {"answer": "Question is too long. Keep it under 1000 characters.", "sources": []}
    try:
        return get_rag_instance().query(message)
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}", "sources": []}
