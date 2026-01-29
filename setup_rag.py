"""
Quick setup script for BatteryLab RAG Copilot
Helps users configure the system
"""

import os
import sys

def check_papers():
    """Check if papers directory exists and has PDFs."""
    papers_dir = "papers"
    if not os.path.exists(papers_dir):
        print(f"❌ Papers directory not found: {papers_dir}")
        return False
    
    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
    if len(pdf_files) == 0:
        print(f"❌ No PDF files found in {papers_dir}")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files in {papers_dir}")
    return True


def check_api_key():
    """Check if OpenAI API key is configured."""
    # Check environment variable
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key found in environment variable")
        return True
    
    # Check Streamlit secrets
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r') as f:
                content = f.read()
                if "OPENAI_API_KEY" in content:
                    print("✅ OpenAI API key found in Streamlit secrets")
                    return True
        except:
            pass
    
    print("❌ OpenAI API key not found")
    print("   Set OPENAI_API_KEY environment variable or add to .streamlit/secrets.toml")
    return False


def check_vector_db():
    """Check if vector database exists."""
    rag_dir = "rag_data"
    if not os.path.exists(rag_dir):
        print(f"❌ Vector database directory not found: {rag_dir}")
        print("   Run: python rag_ingestion.py")
        return False
    
    required_files = ["chunks.json", "faiss.index"]
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(rag_dir, f)):
            missing.append(f)
    
    if missing:
        print(f"❌ Vector database incomplete. Missing: {', '.join(missing)}")
        print("   Run: python rag_ingestion.py")
        return False
    
    print("✅ Vector database found and complete")
    return True


def main():
    """Run setup checks."""
    print("=" * 60)
    print("BatteryLab RAG Copilot - Setup Check")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check papers
    print("1. Checking papers directory...")
    if not check_papers():
        all_ok = False
    print()
    
    # Check API key
    print("2. Checking OpenAI API key...")
    if not check_api_key():
        all_ok = False
    print()
    
    # Check vector database
    print("3. Checking vector database...")
    if not check_vector_db():
        all_ok = False
    print()
    
    print("=" * 60)
    if all_ok:
        print("✅ All checks passed! RAG Copilot is ready to use.")
        print("   Start the app with: streamlit run app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print()
        print("Quick setup:")
        print("1. Ensure PDFs are in papers/ directory")
        print("2. Set OPENAI_API_KEY environment variable or add to .streamlit/secrets.toml")
        print("3. Run: python rag_ingestion.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
