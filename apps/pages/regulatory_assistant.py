from __future__ import annotations

import streamlit as st

from src.rag import build_rag_context


def render() -> None:
    st.title("Regulatory Assistant")
    st.caption("A future-ready RAG interface for policy and compliance support.")
    st.info("This placeholder is ready to be connected to a retrieval pipeline when policy documents are available.")

    prompt = st.text_area("Ask a question about credit policy or regulatory guidance", value="")
    if st.button("Generate Context"):
        st.info(build_rag_context(prompt))
