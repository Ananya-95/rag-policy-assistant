import streamlit as st
from src.pipeline.rag_pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Cached pipeline  (one instance per Streamlit session-server)
# ---------------------------------------------------------------------------

@st.cache_resource
def _pipeline() -> RAGPipeline:
    return RAGPipeline(use_hybrid=True, use_query_rewrite=True, memory_k=5)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Policy RAG v0.3",
        page_icon="\U0001f4cb",
        layout="wide",
    )

    pipeline = _pipeline()

    # ── session-state init ───────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("\U0001f4cb Policy RAG Assistant")
        st.markdown("**Version:** v0.3 – Hybrid + Window Memory + Multi-Query")
        st.divider()

        st.header("How to use")
        st.markdown(
            "1. Put PDFs in `data/Docs/`.\n"
            "2. Run `python main.py index` once to build the index.\n"
            "3. Set `GROQ_API_KEY` (e.g. in `.env`).\n"
            "4. Type your question in the box at the **bottom** of the page."
        )
        st.divider()

        # Retriever info
        st.subheader("\u2699\ufe0f Retrieval mode")
        st.info("Hybrid: BM25 + FAISS + Multi-Query rerank")

        # Memory info
        st.subheader("\U0001f9e0 Memory")
        mem_msgs = len(pipeline.memory)
        turns = mem_msgs // 2
        st.info(f"ConversationBufferWindowMemory – last 5 turns\n\n"
                f"**Active:** {turns}/5 turn(s) in window")

        st.divider()
        if st.button("\U0001f5d1\ufe0f  Clear conversation", use_container_width=True):
            st.session_state.messages = []
            pipeline.clear_history()
            st.rerun()

        # Debug expander: show raw window memory
        if st.session_state.messages:
            with st.expander("🔍 Window memory (debug)", expanded=False):
                for msg in pipeline.get_history():
                    role_label = "🧑 Human" if msg["role"] == "user" else "🤖 Assistant"
                    st.markdown(f"**{role_label}:** {msg['content'][:200]}{'…' if len(msg['content']) > 200 else ''}")

    # ── main area ────────────────────────────────────────────────────────
    st.title("Ask about your policies")
    st.caption(
        "Multi-turn chat · Hybrid retrieval (BM25 + FAISS) · "
        "LLM query rewriter · Multi-query dedup rerank · "
        "Window memory (last 5 turns) · Groq Llama-3"
    )

    # ── render chat history ───────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── chat input ────────────────────────────────────────────────────────
    if prompt := st.chat_input("Type your question here…"):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate reply
        with st.chat_message("assistant"):
            with st.spinner("Rewriting query · Retrieving · Generating…"):
                reply = pipeline.answer(prompt)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

        # Refresh sidebar memory counter without full page reload
        st.rerun()


if __name__ == "__main__":
    main()
