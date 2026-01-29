"""
RAG Backend for BatteryLab Copilot
Handles retrieval and answer generation from research papers
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import streamlit as st

# Configuration
VECTOR_DB_DIR = "rag_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RETRIEVAL_CHUNKS = 5
TEMPERATURE = 0.1
MAX_TOKENS = 500

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


class BatteryLabRAG:
    """RAG system for BatteryLab research papers."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize RAG system.
        
        Parameters:
        -----------
        openai_api_key : str, optional
            OpenAI API key. If None, tries to get from environment or Streamlit secrets.
        """
        # Load embedding model with error handling
        # The model should be cached locally from rag_ingestion.py
        import os
        import time
        
        # Set environment variable to use offline mode if possible
        os.environ.setdefault('HF_HUB_OFFLINE', '0')
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                # Try loading with local_files_only first (avoids HTTP requests)
                if attempt == 0:
                    try:
                        self.embedding_model = SentenceTransformer(
                            EMBEDDING_MODEL,
                            device='cpu',
                            local_files_only=True
                        )
                        break  # Success!
                    except Exception as local_err:
                        # If local_only fails, continue to try with network access
                        last_error = local_err
                        if attempt < max_attempts - 1:
                            time.sleep(1)
                            continue
                
                # Try with network access (will use cache if available)
                self.embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device='cpu',
                    local_files_only=False
                )
                break  # Success!
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # If it's a client closed error, wait longer and retry
                if "client has been closed" in error_str:
                    if attempt < max_attempts - 1:
                        wait_time = 2 * (attempt + 1)  # 2, 4, 6 seconds
                        time.sleep(wait_time)
                        continue
                
                # If it's the last attempt, raise the error
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Failed to load embedding model '{EMBEDDING_MODEL}' after {max_attempts} attempts. "
                        f"Last error: {str(last_error)}. "
                        f"The model should be cached locally from rag_ingestion.py. "
                        f"Try running 'python rag_ingestion.py' again to refresh the model cache, "
                        f"or check your internet connection."
                    )
                
                time.sleep(1)
        
        # Initialize OpenAI client
        if openai_api_key is None:
            # Try to get from Streamlit secrets or environment
            try:
                openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            except:
                openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or add to Streamlit secrets.toml"
            )
        
        # Store API key for potential reinitialization
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Load vector database
        self._load_vector_db()
    
    def _load_vector_db(self):
        """Load FAISS index and chunks metadata."""
        chunks_path = os.path.join(VECTOR_DB_DIR, "chunks.json")
        index_path = os.path.join(VECTOR_DB_DIR, "faiss.index")
        
        if not os.path.exists(chunks_path) or not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Vector database not found. Run rag_ingestion.py first.\n"
                f"Expected files: {chunks_path}, {index_path}"
            )
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Suppress print in Streamlit
        # print(f"Loaded vector database: {len(self.chunks)} chunks, {self.index.ntotal} vectors")
    
    def _retrieve_chunks(self, query: str, k: int = MAX_RETRIEVAL_CHUNKS) -> List[Dict]:
        """
        Retrieve most relevant chunks using MMR (Maximal Marginal Relevance).
        
        Parameters:
        -----------
        query : str
            User query
        k : int
            Number of chunks to retrieve
        
        Returns:
        --------
        List[Dict]
            Retrieved chunks with metadata
        """
        # Encode query with error handling
        try:
            query_embedding = self.embedding_model.encode([query])
        except Exception as e:
            # If encoding fails, try to reinitialize the model
            try:
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                query_embedding = self.embedding_model.encode([query])
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to encode query. Original error: {str(e)}. "
                    f"Reinit error: {str(e2)}"
                )
        
        faiss.normalize_L2(query_embedding)
        query_embedding = query_embedding.astype('float32')
        
        # Retrieve more candidates for MMR
        search_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, search_k)
        
        # MMR: Select diverse chunks
        selected_indices = []
        selected_embeddings = []
        
        # Get embeddings for selected chunks (we'll need to load them)
        # For simplicity, use top-k with diversity via distance threshold
        seen_papers = set()
        for idx in indices[0]:
            if len(selected_indices) >= k:
                break
            
            chunk = self.chunks[idx]
            paper = chunk['paper']
            
            # Prefer diversity: don't take too many from same paper
            paper_count = sum(1 for i in selected_indices if self.chunks[i]['paper'] == paper)
            if paper_count < 2:  # Max 2 chunks per paper
                selected_indices.append(int(idx))
            elif len(selected_indices) < k and len(seen_papers) < len(set(c['paper'] for c in self.chunks)):
                # If we need more and haven't covered all papers, add it
                selected_indices.append(int(idx))
            
            seen_papers.add(paper)
        
        # Return selected chunks
        retrieved_chunks = [self.chunks[i] for i in selected_indices[:k]]
        
        return retrieved_chunks
    
    def _generate_answer(self, query: str, context_chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Generate answer from retrieved context.
        
        Parameters:
        -----------
        query : str
            User query
        context_chunks : List[Dict]
            Retrieved context chunks
        
        Returns:
        --------
        Tuple[str, List[Dict]]
            Generated answer and source citations
        """
        if not context_chunks:
            return "Not covered in my knowledge core.", []
        
        # Build context from chunks
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
        
        # Format context with paper citations instead of "Source X"
        context_lines = []
        for text, chunk in zip(context_parts, context_chunks):
            citation = paper_citations[chunk['paper']]
            context_lines.append(f"[{citation}]\n{text}")
        context = "\n\n".join(context_lines)
        
        # Create system prompt
        system_prompt = """You are BatteryLab Copilot, an AI assistant that answers questions about battery research based ONLY on the provided research papers.

CRITICAL RULES:
1. Answer ONLY using information from the provided context. Do not use any external knowledge.
2. If the context does not contain enough information to answer the question, respond: "Not covered in my knowledge core."
3. Be concise, factual, and research-oriented.
4. Cite sources by paper author and year (e.g., "According to Severson et al. (2019)..."). Use the exact citation format shown in brackets before each context section.
5. Do not hallucinate or make up information.
6. If asked about something not in the papers, clearly state it's not covered.

Format your answer clearly and cite sources when appropriate."""

        user_prompt = f"""Context from research papers:

{context}

Question: {query}

Answer based ONLY on the context above:"""

        try:
            # Call OpenAI API with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",  # Use cost-effective model
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS
                    )
                    
                    answer = response.choices[0].message.content.strip()
                    
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
                    error_str = str(e).lower()
                    
                    # Handle specific error types
                    if "insufficient_quota" in error_str or "429" in str(e) or "quota" in error_str:
                        return (
                            "⚠️ **OpenAI API Quota Exceeded**\n\n"
                            "Your OpenAI API account has exceeded its quota or billing limit. "
                            "Please check your OpenAI account billing and usage:\n"
                            "- Visit https://platform.openai.com/account/billing\n"
                            "- Add payment method or upgrade your plan\n"
                            "- Check your usage limits\n\n"
                            "Once your quota is restored, the Research Copilot will work again.",
                            []
                        )
                    elif "client has been closed" in error_str:
                        # Reinitialize OpenAI client if it was closed
                        try:
                            self.openai_client = OpenAI(api_key=self.openai_api_key)
                        except Exception as reinit_error:
                            if attempt == max_retries - 1:
                                return f"Error generating answer: Failed to reinitialize client. {str(e)}", []
                    elif attempt == max_retries - 1:
                        # Format error message nicely
                        error_msg = str(e)
                        if "error" in error_str and "message" in error_str:
                            # Try to extract the actual error message
                            try:
                                import json
                                if "{" in error_msg:
                                    error_dict = eval(error_msg.split("{")[-1].split("}")[0] + "}")
                                    if "error" in error_dict and "message" in error_dict["error"]:
                                        error_msg = error_dict["error"]["message"]
                            except:
                                pass
                        return f"Error generating answer: {error_msg}", []
                    
                    import time
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            
        except Exception as e:
            return f"Error generating answer: {str(e)}", []
    
    def query(self, message: str) -> Dict:
        """
        Process user query and return answer with sources.
        
        Parameters:
        -----------
        message : str
            User query
        
        Returns:
        --------
        Dict
            {
                "answer": str,
                "sources": List[Dict]
            }
        """
        # Retrieve relevant chunks
        retrieved_chunks = self._retrieve_chunks(message, k=MAX_RETRIEVAL_CHUNKS)
        
        # Generate answer
        answer, sources = self._generate_answer(message, retrieved_chunks)
        
        return {
            "answer": answer,
            "sources": sources
        }


# Global RAG instance (lazy loading)
_rag_instance = None


def get_rag_instance() -> BatteryLabRAG:
    """Get or create global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = BatteryLabRAG()
    return _rag_instance


def chat_api(message: str) -> Dict:
    """
    API endpoint for chat queries.
    
    Parameters:
    -----------
    message : str
        User message
    
    Returns:
    --------
    Dict
        {
            "answer": str,
            "sources": List[Dict]
        }
    """
    # Basic input validation
    if not message or len(message.strip()) == 0:
        return {
            "answer": "Please provide a question.",
            "sources": []
        }
    
    if len(message) > 1000:
        return {
            "answer": "Question is too long. Please keep it under 1000 characters.",
            "sources": []
        }
    
    try:
        rag = get_rag_instance()
        return rag.query(message)
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": []
        }
