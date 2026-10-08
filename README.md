# ⚡ Enterprise RAG Knowledge Search Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v1.0+-blue.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-red.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Llama_3.2_(Local)-black.svg)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade, 100% private Retrieval-Augmented Generation (RAG) platform that enables semantic search and conversational QA over enterprise documents (PDF/TXT). Built using **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and a local **Llama 3.2** LLM served via an interactive **Streamlit** dashboard.

---

## 🌟 Key Features

* **🔒 100% Privacy & Zero API Cost:** Runs entirely locally using Ollama (`llama3.2`) and local HuggingFace embeddings—no data is ever transmitted to third-party APIs.
* **📄 Automated Document Ingestion:** Supports dense PDFs and TXT files with recursive character text chunking (`chunk_size=500`, `overlap=50`).
* **⚡ Vector Similarity Search:** Generates high-dimensional vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` stored in persistent ChromaDB.
* **🔍 Explainable AI & Source Provenance:** Every response includes inspectable raw context chunks retrieved from the knowledge base to prevent hallucinations.
* **💬 Conversational UI:** Built-in Streamlit chat session management with real-time indexing status indicators.

---

## 📐 System Architecture

```text
[ Document Upload (PDF/TXT) ]
              │
              ▼
[ Recursive Text Chunking (LangChain) ]
              │
              ▼
[ HuggingFace MiniLM Embeddings ] ──► [ Persistent Vector DB (ChromaDB) ]
                                                       │
                                                       ▼
[ User Query ] ──────────────────────────► [ Similarity Search (k=3) ]
                                                       │
                                                       ▼
[ Synthesized Answer ] ◄── [ Ollama Llama 3.2 ] ◄── [ Relevant Context Chunks ]
