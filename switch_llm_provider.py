"""
Quick script to switch LLM providers for BatteryLab RAG
"""

import os
import sys

def set_provider(provider: str):
    """Set the LLM provider."""
    valid_providers = ["openai", "ollama", "huggingface"]
    
    if provider.lower() not in valid_providers:
        print(f"❌ Invalid provider: {provider}")
        print(f"Valid options: {', '.join(valid_providers)}")
        return False
    
    # Set environment variable
    os.environ["LLM_PROVIDER"] = provider.lower()
    
    print(f"✅ LLM Provider set to: {provider.upper()}")
    print(f"\nNext steps:")
    
    if provider.lower() == "ollama":
        print("1. Install Ollama: https://ollama.com/download")
        print("2. Run: ollama pull llama3.2")
        print("3. Ensure Ollama is running")
        print("4. Update app.py: from rag_backend_multi import chat_api")
        print("5. Restart Streamlit")
    elif provider.lower() == "huggingface":
        print("1. Install: pip install transformers torch")
        print("2. Update app.py: from rag_backend_multi import chat_api")
        print("3. Restart Streamlit")
        print("   (First run will download model)")
    else:  # openai
        print("1. Set OPENAI_API_KEY in .streamlit/secrets.toml")
        print("2. Update app.py: from rag_backend_multi import chat_api")
        print("3. Restart Streamlit")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python switch_llm_provider.py <provider>")
        print("\nProviders:")
        print("  ollama       - Free local LLM (recommended)")
        print("  huggingface  - Free local LLM (slower)")
        print("  openai       - Paid cloud API")
        print("\nExample: python switch_llm_provider.py ollama")
        sys.exit(1)
    
    provider = sys.argv[1]
    set_provider(provider)
