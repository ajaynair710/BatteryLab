"""
RAG Backend for BatteryLab Copilot
Handles retrieval and answer generation from research papers using Ollama
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import streamlit as st

# Configuration
VECTOR_DB_DIR = "rag_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RETRIEVAL_CHUNKS = 3  # Reduced for faster processing
TEMPERATURE = 0.1
MAX_TOKENS = 300  # Reduced for faster responses
OLLAMA_TIMEOUT = 60  # 60 second timeout

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

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class BatteryLabRAG:
    """RAG system for BatteryLab research papers using Ollama."""
    
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize RAG system with Ollama.
        
        Parameters:
        -----------
        model : str, optional
            Ollama model name (default: llama3.2)
        base_url : str, optional
            Ollama base URL (default: http://localhost:11434)
        """
        self.model = model or OLLAMA_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        
        # Load embedding model
        self._load_embedding_model()
        
        # Initialize Ollama
        self._init_ollama()
        
        # Load vector database
        self._load_vector_db()
    
    def _load_embedding_model(self):
        """Load embedding model with error handling."""
        import time
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if attempt == 0:
                    try:
                        self.embedding_model = SentenceTransformer(
                            EMBEDDING_MODEL,
                            device='cpu',
                            local_files_only=True
                        )
                        return
                    except:
                        pass
                
                self.embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device='cpu',
                    local_files_only=False
                )
                return
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Failed to load embedding model: {str(e)}"
                    )
                time.sleep(1)
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            from ollama import chat, list
            self.chat = chat
            self.list_models = list
        except ImportError:
            raise ImportError(
                "Install ollama Python package: pip install ollama\n"
                "Also ensure Ollama service is installed: https://ollama.com/download"
            )
        
        # Check if Ollama is running and model is available
        self._check_ollama_running()
    
    def _check_ollama_running(self):
        """Check if Ollama is running and model is available."""
        try:
            # Try to list models to verify Ollama is running
            models_response = self.list_models()
            
            # Extract model names - handle both dict and Model object formats
            model_names = []
            if hasattr(models_response, 'models'):
                for m in models_response.models:
                    if hasattr(m, 'model'):
                        # Model object format
                        model_names.append(m.model)
                    elif isinstance(m, dict):
                        # Dictionary format
                        model_names.append(m.get('name', m.get('model', '')))
            
            # Check if our model is available (handle :latest tag)
            model_found = False
            for name in model_names:
                # Check if model name matches or starts with our model name
                if name == self.model or name.startswith(self.model + ':') or self.model in name:
                    model_found = True
                    # Update self.model to use the actual name if it has :latest tag
                    if ':' in name and ':' not in self.model:
                        self.model = name.split(':')[0]  # Use base name without tag
                    break
            
            if not model_found:
                available = ', '.join(model_names[:3]) if model_names else 'none'
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Available models: {available}\n\n"
                    f"Download it with: ollama pull {self.model}"
                )
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "refused" in error_msg.lower():
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}.\n"
                    "Make sure Ollama is installed and running:\n"
                    "1. Install: https://ollama.com/download\n"
                    "2. Start Ollama service\n"
                    f"3. Run: ollama pull {self.model}"
                )
            raise
    
    def _load_vector_db(self):
        """Load FAISS index and chunks metadata."""
        chunks_path = os.path.join(VECTOR_DB_DIR, "chunks.json")
        index_path = os.path.join(VECTOR_DB_DIR, "faiss.index")
        
        if not os.path.exists(chunks_path) or not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Vector database not found. Run rag_ingestion.py first.\n"
                f"Expected files: {chunks_path}, {index_path}"
            )
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        self.index = faiss.read_index(index_path)
    
    def _retrieve_chunks(self, query: str, k: int = MAX_RETRIEVAL_CHUNKS) -> List[Dict]:
        """Retrieve most relevant chunks."""
        try:
            query_embedding = self.embedding_model.encode([query])
        except Exception as e:
            raise RuntimeError(f"Failed to encode query: {str(e)}")
        
        faiss.normalize_L2(query_embedding)
        query_embedding = query_embedding.astype('float32')
        
        search_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)
        
        # MMR: Select diverse chunks
        selected_indices = []
        seen_papers = set()
        
        for idx in indices[0]:
            if len(selected_indices) >= k:
                break
            
            chunk = self.chunks[idx]
            paper = chunk['paper']
            
            paper_count = sum(1 for i in selected_indices if self.chunks[i]['paper'] == paper)
            if paper_count < 2:
                selected_indices.append(int(idx))
            elif len(selected_indices) < k:
                selected_indices.append(int(idx))
            
            seen_papers.add(paper)
        
        return [self.chunks[i] for i in selected_indices[:k]]
    
    def _generate_answer(self, query: str, context_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """Generate answer from retrieved context using Ollama."""
        if not context_chunks:
            return "Not covered in my knowledge core.", []
        
        # Build context
        context_parts = []
        sources = []
        paper_citations = {}  # Map paper names to citations
        
        for chunk in context_chunks:
            paper_name = chunk['paper']
            context_parts.append(chunk['text'])
            sources.append({
                'paper': paper_name,
                'section': chunk['section']
            })
            # Build citation mapping
            if paper_name not in paper_citations:
                paper_citations[paper_name] = get_paper_citation(paper_name)
        
        # Truncate context if too long (limit to ~1500 chars per chunk for faster processing)
        truncated_parts = []
        for text, chunk in zip(context_parts, context_chunks):
            citation = paper_citations[chunk['paper']]
            if len(text) > 1500:
                text = text[:1500] + "..."
            truncated_parts.append(f"[{citation}]\n{text}")
        
        context = "\n\n".join(truncated_parts)
        
        # Shorter, more focused prompts for faster processing
        system_prompt = (
            "Answer questions about battery research using ONLY the provided context. "
            "Be concise and factual. Cite sources by paper author and year (e.g., 'Severson et al. (2019)'). "
            "Use the exact citation format shown in brackets before each context section. "
            "If information isn't in the context, say 'Not covered in my knowledge core.'"
        )

        user_prompt = (
            f"Context:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        try:
            # Use Ollama to generate answer
            # Reduced context and tokens for faster processing
            response = self.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': TEMPERATURE,
                    'num_predict': MAX_TOKENS,
                    'num_ctx': 2048  # Limit context window for faster processing
                }
            )
            
            # Handle both dict and object response formats
            if hasattr(response, 'message'):
                answer = response.message.content.strip()
            elif isinstance(response, dict):
                answer = response['message']['content'].strip()
            else:
                answer = str(response).strip()
            
            # Remove duplicate sources
            unique_sources = []
            seen = set()
            for source in sources:
                key = (source['paper'], source['section'])
                if key not in seen:
                    unique_sources.append(source)
                    seen.add(key)
            
            return answer, unique_sources
            
        except Exception as e:
            error_msg = str(e)
            if "Connection" in error_msg or "refused" in error_msg.lower():
                return (
                    f"⚠️ **Ollama Connection Error**\n\n"
                    f"Cannot connect to Ollama at {self.base_url}.\n"
                    "Make sure Ollama is running:\n"
                    "1. Start Ollama service\n"
                    f"2. Verify model is available: ollama list\n"
                    f"3. If missing, run: ollama pull {self.model}",
                    []
                )
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                return (
                    f"⚠️ **Model Not Found**\n\n"
                    f"Model '{self.model}' is not available.\n"
                    f"Download it with: ollama pull {self.model}",
                    []
                )
            # Check for timeout-like errors
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                return (
                    f"⚠️ **Request Timeout**\n\n"
                    f"The request took too long. Try:\n"
                    f"- Asking a shorter, more specific question\n"
                    f"- Restarting Ollama service\n"
                    f"- Using a smaller model",
                    []
                )
            return f"Error generating answer: {error_msg}", []
    
    def query(self, message: str) -> Dict:
        """Process user query and return answer with sources."""
        retrieved_chunks = self._retrieve_chunks(message, k=MAX_RETRIEVAL_CHUNKS)
        answer, sources = self._generate_answer(message, retrieved_chunks)
        
        return {
            "answer": answer,
            "sources": sources
        }


# Global RAG instance
_rag_instance = None


def get_rag_instance() -> BatteryLabRAG:
    """Get or create global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = BatteryLabRAG()
    return _rag_instance


def chat_api(message: str) -> Dict:
    """API endpoint for chat queries."""
    if not message or len(message.strip()) == 0:
        return {"answer": "Please provide a question.", "sources": []}
    
    if len(message) > 1000:
        return {"answer": "Question is too long. Please keep it under 1000 characters.", "sources": []}
    
    try:
        rag = get_rag_instance()
        return rag.query(message)
    except Exception as e:
        return {"answer": f"Error processing query: {str(e)}", "sources": []}
