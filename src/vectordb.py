import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DB_PATH = "./.chroma_db"
_retriever = None

def init_retriever(doc_splits):
    """Explicit initialization used by main.py ingestion pipelines."""
    global _retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=doc_splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever

def get_retriever():
    """Lazy-loading retrieval node entry point used by LangGraph Studio."""
    global _retriever
    if _retriever is not None:
        return _retriever

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. Optimal Path: If database files exist on disk, attach immediately
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return _retriever
        
    # 2. Fallback Path: Build the directory structure if Studio runs cold
    from src.documents import load_web_page
    from src.chunks import split_docs

    print("⚠️ Local DB cache not found. Re-indexing documents for Studio session...")
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    docs = [load_web_page(url) for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    doc_splits = split_docs(docs_list)

    return init_retriever(doc_splits)