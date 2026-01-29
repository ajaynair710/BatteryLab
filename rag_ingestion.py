"""
Offline PDF Ingestion Pipeline for BatteryLab RAG System
Processes 7 research papers and creates a FAISS vector database
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import PyPDF2
import pdfplumber

# Configuration
PAPERS_DIR = "papers"
VECTOR_DB_DIR = "rag_data"
CHUNK_SIZE = 700  # tokens (approximately)
CHUNK_OVERLAP = 100  # tokens
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast, good quality


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[Dict]]:
    """
    Extract text from PDF with section information.
    
    Returns:
    --------
    Tuple[str, List[Dict]]
        Full text and list of sections with page numbers
    """
    full_text = ""
    sections = []
    
    try:
        # Try pdfplumber first (better for complex PDFs)
        with pdfplumber.open(pdf_path) as pdf:
            current_section = None
            page_num = 0
            
            for page in pdf.pages:
                page_num += 1
                page_text = page.extract_text()
                
                if page_text:
                    full_text += page_text + "\n"
                    
                    # Try to detect section headers (lines in all caps or with numbers)
                    lines = page_text.split('\n')
                    for line in lines[:5]:  # Check first few lines of page
                        line_clean = line.strip()
                        if (len(line_clean) > 3 and len(line_clean) < 100 and
                            (line_clean.isupper() or 
                             (line_clean[0].isdigit() and '.' in line_clean[:10]))):
                            current_section = line_clean
                            sections.append({
                                'section': current_section,
                                'page': page_num,
                                'start_char': len(full_text) - len(page_text)
                            })
    except Exception as e:
        print(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")
        # Fallback to PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
    
    return full_text, sections


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split text into overlapping chunks.
    
    Parameters:
    -----------
    text : str
        Input text
    chunk_size : int
        Target chunk size in tokens (approximate)
    overlap : int
        Overlap size in tokens
    
    Returns:
    --------
    List[Dict]
        List of chunks with metadata
    """
    # Simple token approximation: ~4 characters per token
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4
    
    chunks = []
    start = 0
    chunk_idx = 0
    
    while start < len(text):
        end = start + char_chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings near the end
            for i in range(end, max(start + char_chunk_size - 200, start), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) > 50:  # Only add substantial chunks
            chunks.append({
                'text': chunk_text,
                'start': start,
                'end': end,
                'chunk_idx': chunk_idx
            })
            chunk_idx += 1
        
        # Move start position with overlap
        start = end - char_overlap
        if start >= len(text):
            break
    
    return chunks


def process_papers() -> Tuple[List[Dict], np.ndarray]:
    """
    Process all PDFs in papers directory and create embeddings.
    
    Returns:
    --------
    Tuple[List[Dict], np.ndarray]
        List of chunk metadata and embedding matrix
    """
    papers_dir = Path(PAPERS_DIR)
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers directory not found: {PAPERS_DIR}")
    
    # Initialize embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    all_chunks = []
    all_embeddings = []
    
    # Process each PDF
    pdf_files = sorted(list(papers_dir.glob("*.pdf")))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        paper_name = pdf_path.stem
        print(f"\nProcessing: {paper_name}")
        
        # Extract text
        full_text, sections = extract_text_from_pdf(str(pdf_path))
        
        if not full_text.strip():
            print(f"  Warning: No text extracted from {paper_name}")
            continue
        
        # Chunk text
        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  Created {len(chunks)} chunks")
        
        # Create embeddings for each chunk
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = model.encode(chunk_texts, show_progress_bar=True)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            # Find which section this chunk belongs to
            section_name = "Introduction"  # Default
            for sec in reversed(sections):
                if chunk['start'] >= sec['start_char']:
                    section_name = sec['section']
                    break
            
            chunk_metadata = {
                'paper': paper_name,
                'section': section_name,
                'chunk_idx': chunk['chunk_idx'],
                'text': chunk['text'],
                'start_char': chunk['start'],
                'end_char': chunk['end']
            }
            all_chunks.append(chunk_metadata)
            all_embeddings.append(embeddings[i])
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    print(f"Embedding dimension: {len(all_embeddings[0]) if all_embeddings else 0}")
    
    return all_chunks, np.array(all_embeddings)


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Create FAISS index from embeddings.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embedding matrix (n_chunks, embedding_dim)
    
    Returns:
    --------
    faiss.Index
        FAISS index
    """
    dimension = embeddings.shape[1]
    
    # Use L2 distance (Inner Product for cosine similarity with normalized embeddings)
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    
    index.add(embeddings.astype('float32'))
    
    print(f"FAISS index created with {index.ntotal} vectors")
    return index


def save_vector_db(chunks: List[Dict], index: faiss.Index, embeddings: np.ndarray):
    """
    Save vector database to disk.
    
    Parameters:
    -----------
    chunks : List[Dict]
        Chunk metadata
    index : faiss.Index
        FAISS index
    embeddings : np.ndarray
        Embedding matrix
    """
    # Create directory
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    # Save chunks metadata
    chunks_path = os.path.join(VECTOR_DB_DIR, "chunks.json")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Save FAISS index
    index_path = os.path.join(VECTOR_DB_DIR, "faiss.index")
    faiss.write_index(index, index_path)
    
    # Save embeddings (for potential re-indexing)
    embeddings_path = os.path.join(VECTOR_DB_DIR, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    
    print(f"\nVector database saved to {VECTOR_DB_DIR}/")
    print(f"  - Chunks: {chunks_path}")
    print(f"  - Index: {index_path}")
    print(f"  - Embeddings: {embeddings_path}")


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("BatteryLab RAG Ingestion Pipeline")
    print("=" * 60)
    
    # Process papers
    chunks, embeddings = process_papers()
    
    if len(chunks) == 0:
        print("Error: No chunks created. Check PDF files.")
        return
    
    # Create FAISS index
    print("\nCreating FAISS index...")
    index = create_faiss_index(embeddings)
    
    # Save vector database
    save_vector_db(chunks, index, embeddings)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
