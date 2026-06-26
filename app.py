from flask import Flask, render_template, request, jsonify
import os
import sys
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Project imports
from loadDoc import load_document
from text_splitter import split_documents
from embeddings import OllamaEmbeddings
from vector_store import get_collection, build_vector_store, CHROMA_PATH
from query import OllamaLLM, ChromaDBRetriever, format_docs

app = Flask(__name__)

# Initialize components globally
emb_model = None
collection = None
retriever = None
llm = None
rag_chain = None
document_loaded = False
chunk_count = 0

def init_rag():
    global emb_model, collection, retriever, llm, rag_chain, document_loaded, chunk_count
    try:
        emb_model = OllamaEmbeddings()
        
        # Check if DB exists, if not build it from rules.pdf
        if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
            print("Vector store not found. Creating from rules.pdf...")
            pdf_path = "./docs/rules.pdf"
            if not os.path.exists(pdf_path):
                print(f"Error: {pdf_path} not found.")
                return False
                
            docs = load_document(pdf_path)
            chunks = split_documents(docs)
            collection = build_vector_store(chunks, emb_model)
            chunk_count = len(chunks)
        else:
            print("Loading existing vector store...")
            collection = get_collection()
            chunk_count = collection.count()
            
        retriever = ChromaDBRetriever(collection=collection, embeddings_model=emb_model, k=3)
        llm = OllamaLLM(model_name="gemma3:4b")
        
        prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful customer service assistant. Use only the following retrieved pieces of context to answer the question.\n"
            "If you don't know the answer, say that you don't know. Keep your answer professional and concise.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        document_loaded = True
        print("RAG System successfully initialized!")
        return True
    except Exception as e:
        print(f"Error during RAG initialization: {e}", file=sys.stderr)
        return False

# Initialize RAG on startup
init_rag()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    global document_loaded, chunk_count
    return jsonify({
        "status": "ready" if document_loaded else "error",
        "document_loaded": document_loaded,
        "chunk_count": int(chunk_count),
        "model_name": "gemma3:4b",
        "embedding_model": "qwen3-embedding:0.6b"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    global rag_chain, document_loaded
    if not document_loaded:
        if not init_rag():
            return jsonify({"error": "RAG system not initialized. Please check backend logs."}), 500
            
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing message in request"}), 400
        
    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
        
    try:
        response = rag_chain.invoke(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Error invoking RAG chain: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
