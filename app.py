import os
import streamlit as st
from rag_pipeline import process_and_store_document, query_rag_engine

st.set_page_config(page_title="Enterprise RAG Engine", page_icon="⚡", layout="wide")

st.title("⚡ Enterprise RAG Knowledge Search Engine")
st.caption("Built with LangChain, ChromaDB, HuggingFace Embeddings & Local Ollama LLM")

st.sidebar.header("📂 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT File", type=["pdf", "txt"])

if uploaded_file is not None:
    os.makedirs("./docs", exist_ok=True)
    save_path = os.path.join("./docs", uploaded_file.name)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.sidebar.button("Index Document"):
        with st.spinner("Processing vector embeddings & saving to ChromaDB..."):
            chunks_count = process_and_store_document(save_path)
            st.sidebar.success(f"Successfully indexed into {chunks_count} vector chunks!")

st.subheader("💬 Query Your Enterprise Knowledge Base")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User query input
if prompt := st.chat_input("Ask a question about your uploaded document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context chunks and querying LLM..."):
            answer, sources = query_rag_engine(prompt)
            st.markdown(answer)
            
            if sources:
                with st.expander("🔍 View Retrieved Source Chunks"):
                    for idx, doc in enumerate(sources):
                        st.write(f"**Chunk {idx+1}:** {doc.page_content[:250]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
