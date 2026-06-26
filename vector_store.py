import os
import shutil
import chromadb
from typing import List, Any
from langchain_core.documents import Document

CHROMA_PATH = "./chroma_db"

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(collection_name="company_policy"):
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)

def build_vector_store(chunks: List[Document], embeddings_model: Any, collection_name="company_policy"):
    # Clear existing database directory to avoid duplicate/stale document insertions
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            print(f"Warning: Could not remove old database directory: {e}")
        
    collection = get_collection(collection_name)
    
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    ids = [f"doc_{idx}" for idx in range(len(chunks))]
    
    print(f"Generating embeddings and storing {len(chunks)} chunks in ChromaDB...")
    # Embed the texts
    embeddings = embeddings_model.embed_documents(texts)
    
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Successfully stored in collection '{collection_name}' at '{CHROMA_PATH}'.")
    return collection

if __name__ == "__main__":
    # Test full ingestion pipeline
    from loadDoc import load_document
    from text_splitter import split_documents
    from embeddings import OllamaEmbeddings
    
    print("Running vector store build test...")
    docs = load_document("./docs/rules.pdf")
    chunks = split_documents(docs)
    emb_model = OllamaEmbeddings()
    build_vector_store(chunks, emb_model)
