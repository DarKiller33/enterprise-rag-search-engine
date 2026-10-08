import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

DB_DIR = "./chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return None

def process_and_store_document(file_path):
    # Load PDF or TXT
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    # Generate embeddings and store in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return len(chunks)

def query_rag_engine(query_text, model_name="llama3.2"):
    vectorstore = get_vectorstore()
    if not vectorstore:
        return "No documents found in knowledge base. Please upload and index a document first.", []
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model=model_name)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    response = qa_chain.invoke({"query": query_text})
    return response["result"], response.get("source_documents", [])
