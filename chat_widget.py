"""
Chat Widget Component for BatteryLab Streamlit App
Provides a chat interface for the RAG-powered copilot
"""

import streamlit as st
from typing import List, Dict
from rag_backend import chat_api


def init_chat_state():
    """Initialize chat session state for RAG copilot."""
    if 'rag_chat_history' not in st.session_state:
        st.session_state.rag_chat_history = []
    if 'rag_chat_initialized' not in st.session_state:
        st.session_state.rag_chat_initialized = True


def render_chat_widget():
    """Render the chat widget in Streamlit."""
    init_chat_state()
    
    st.markdown("### 🔋 BatteryLab Copilot")
    st.markdown("Ask questions about battery research based on our knowledge base of 7 research papers.")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, msg in enumerate(st.session_state.rag_chat_history):
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(msg['content'])
                    
                    # Display sources if available
                    if 'sources' in msg and msg['sources']:
                        with st.expander("📚 Sources", expanded=False):
                            for source in msg['sources']:
                                st.write(f"**{source['paper']}** - {source['section']}")
    
    # Chat input
    user_input = st.chat_input("Ask a question about battery research...")
    
    if user_input:
        # Add user message to history
        st.session_state.rag_chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat_api(user_input)
                    answer = response.get('answer', 'No answer generated.')
                    sources = response.get('sources', [])
                    
                    st.write(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sources", expanded=True):
                            for source in sources:
                                st.write(f"**{source['paper']}** - {source['section']}")
                    
                    # Add assistant response to history
                    st.session_state.rag_chat_history.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.rag_chat_history.append({
                        'role': 'assistant',
                        'content': error_msg
                    })
        
        # Rerun to update chat display
        st.rerun()


def render_chat_sidebar():
    """Render chat controls in sidebar."""
    st.sidebar.markdown("### Chat Controls")
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.rag_chat_history = []
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About BatteryLab Copilot**")
    st.sidebar.markdown("""
    This copilot answers questions based on 7 research papers:
    - Severson et al. (Nature Energy, 2019)
    - Early cycles analysis
    - ICA for SOH at different temperatures
    - SEI model
    - SoH with C-rate empirical
    - Arrhenius-derived empirical
    - Wang et al.
    
    The system uses RAG (Retrieval-Augmented Generation) to provide
    grounded, citation-aware answers.
    """)
