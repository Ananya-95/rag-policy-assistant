import streamlit as st

from src.pipeline.rag_pipeline import RAGPipeline


@st.cache_resource
def _pipeline() -> RAGPipeline:
    return RAGPipeline(use_hybrid=True, use_query_rewrite=True, memory_k=5)


def _render_sources(sources) -> None:
    if not sources:
        return
    with st.expander("📎 Sources", expanded=True):
        for i, src in enumerate(sources, 1):
            page = f" · page {src.page}" if src.page is not None else ""
            st.markdown(f"**[{i}] {src.filename}**{page}")
            st.caption(src.snippet)


def main() -> None:
    st.set_page_config(
        page_title="Policy RAG v0.3",
        page_icon="\U0001f4cb",
        layout="wide",
    )

    pipeline = _pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("\U0001f4cb Policy RAG Assistant")
        st.markdown("**Version:** v0.3 – Hybrid + Memory + Citations")
        st.divider()

        st.header("How to use")
        st.markdown(
            "1. Put PDFs in `data/Docs/`.\n"
            "2. Run `python main.py index` once.\n"
            "3. Set `GROQ_API_KEY` in `.env` or HF Secrets.\n"
            "4. Ask questions below."
        )
        st.divider()

        st.subheader("\u2699\ufe0f Retrieval mode")
        st.info("Hybrid: BM25 + FAISS + multi-query rerank")

        st.subheader("\U0001f9e0 Memory")
        turns = len(pipeline.memory) // 2
        st.info(f"Window memory – last 5 turns\n\n**Active:** {turns}/5 turn(s)")

        st.divider()
        if st.button("\U0001f5d1\ufe0f  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            pipeline.clear_history()
            st.rerun()

        if st.session_state.messages:
            with st.expander("🔍 Window memory (debug)", expanded=False):
                for msg in pipeline.get_history():
                    role_label = "🧑 Human" if msg["role"] == "user" else "🤖 Assistant"
                    text = msg["content"][:200]
                    if len(msg["content"]) > 200:
                        text += "…"
                    st.markdown(f"**{role_label}:** {text}")

    st.title("Ask about your policies")
    st.caption(
        "Hybrid retrieval · query rewriting · multi-query · citations · Groq Llama-3.3"
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    if prompt := st.chat_input("Type your question here…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Rewriting query · Retrieving · Generating…"):
                result = pipeline.answer(prompt)
            st.markdown(result.answer)
            _render_sources(result.sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources,
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
