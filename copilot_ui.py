# Shared BatteryLAB Copilot UI (RAG-based). Used by the Copilot page.

import os
import streamlit as st

PAPER_METADATA = {
    "Severson_NatureEnergy_2019": {
        "title": "Data-driven prediction of battery cycle life before capacity degradation",
        "author": "Severson et al.",
        "year": "2019",
        "journal": "Nature Energy"
    },
    "Early cycles": {"title": "Early cycles", "author": "Unknown", "year": "Unknown"},
    "Arrhneius derived emp on Severson": {"title": "Arrhenius-derived empirical model", "author": "Unknown", "year": "Unknown"},
    "ICA for SOH at Diff Temp": {"title": "ICA for State of Health at Different Temperatures", "author": "Unknown", "year": "Unknown"},
    "SEI model": {"title": "SEI Model", "author": "Unknown", "year": "Unknown"},
    "SoH with C rate Emp": {"title": "State of Health with C-rate Empirical Model", "author": "Unknown", "year": "Unknown"},
    "wang et al": {"title": "Wang et al.", "author": "Wang et al.", "year": "Unknown"},
}


def _get_paper_citation(paper_name: str) -> str:
    m = PAPER_METADATA.get(paper_name, {"title": paper_name, "author": "Unknown", "year": "Unknown"})
    if m.get("journal"):
        return f"{m['author']} ({m['year']}), \"{m['title']}\", {m['journal']}"
    return f"{m['author']} ({m['year']}), \"{m['title']}\""


def _copilot_add(role: str, text: str, sources: list = None):
    msg = {"role": role, "text": text}
    if sources:
        msg["sources"] = sources
    st.session_state.chat_history.append(msg)


def render_copilot():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "latest_design" not in st.session_state:
        st.session_state.latest_design = None
    if "latest_analytics" not in st.session_state:
        st.session_state.latest_analytics = None

    st.caption("Ask questions about battery research from our knowledge base of 7 papers.")

    rag_available = False
    backend = (os.getenv("RAG_BACKEND") or "").strip().lower()
    try:
        if getattr(st, "secrets", None) and st.secrets.get("RAG_BACKEND"):
            backend = str(st.secrets.get("RAG_BACKEND", "")).strip().lower()
    except Exception:
        pass
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    try:
        if getattr(st, "secrets", None) and st.secrets.get("OPENAI_API_KEY"):
            has_openai = True
    except Exception:
        pass
    use_ollama = backend == "ollama" or (not backend and not has_openai)
    use_llamacpp = backend == "llamacpp" or bool(os.getenv("HF_LLAMA_REPO_ID")) or (getattr(st, "secrets", None) and st.secrets.get("HF_LLAMA_REPO_ID"))
    try:
        if use_ollama:
            from rag_backend_multi import chat_api
            rag_available = True
        elif use_llamacpp:
            from rag_backend_llamacpp import chat_api
            rag_available = True
        else:
            from rag_backend import chat_api
            rag_available = True
    except Exception as e:
        st.warning("RAG Copilot not available.")
        st.caption(str(e))
        st.info("Free (no OpenAI): Install Ollama, run `ollama pull llama3.2`, `pip install ollama`, set RAG_BACKEND=ollama. See RAG_FREE.md.")

    for msg in st.session_state.chat_history[-12:]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg.get("text", msg.get("content", "")))
        else:
            with st.chat_message("assistant"):
                st.write(msg.get("text", msg.get("content", "")))
                if "sources" in msg and msg["sources"]:
                    with st.expander("Sources", expanded=False):
                        seen = set()
                        for s in msg["sources"]:
                            p = s.get("paper", "Unknown")
                            if p not in seen:
                                seen.add(p)
                                st.write(f"**{_get_paper_citation(p)}**")

    user_text = st.chat_input("Ask about battery research…", key="copilot_input")
    if user_text:
        _copilot_add("user", user_text.strip())
        if rag_available:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        r = chat_api(user_text.strip())
                        ans = r.get("answer", "No answer generated.")
                        src = r.get("sources", [])
                        st.write(ans)
                        if src:
                            with st.expander("Sources", expanded=True):
                                seen = set()
                                for s in src:
                                    p = s.get("paper", "Unknown")
                                    if p not in seen:
                                        seen.add(p)
                                        st.write(f"**{_get_paper_citation(p)}**")
                        _copilot_add("assistant", ans, sources=src)
                    except Exception as e:
                        st.error(str(e))
                        _copilot_add("assistant", f"Error: {e}")
        else:
            with st.chat_message("assistant"):
                reply = "Set up a free backend (Ollama or llama-cpp) or OPENAI_API_KEY. See RAG_FREE.md."
                st.write(reply)
                _copilot_add("assistant", reply)
        st.rerun()
