import os
import sys
import ollama
from typing import Any, List, Optional
from pydantic import Field
from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import modules from the project
from loadDoc import load_document
from text_splitter import split_documents
from embeddings import OllamaEmbeddings
from vector_store import get_collection, build_vector_store, CHROMA_PATH

# 1. Custom LangChain LLM Wrapper for Ollama
class OllamaLLM(LLM):
    model_name: str = "gemma3:4b"

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"stop": stop} if stop else None
            )
            return response['message']['content']
        except Exception as e:
            return f"Error querying Ollama model '{self.model_name}': {e}"

# 2. Custom LangChain Retriever Wrapper for ChromaDB
class ChromaDBRetriever(BaseRetriever):
    collection: Any = Field(exclude=True)
    embeddings_model: Embeddings = Field(exclude=True)
    k: int = 3

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:

        query_vector = self.embeddings_model.embed_query(query)
        # Search the ChromaDB collection
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.k
        )
        docs = []
        if results and 'documents' in results and results['documents']:
            
            for text in results['documents'][0]:
                docs.append(Document(page_content=text))
        return docs

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("=== The bot is at your service ===")
    
    emb_model = OllamaEmbeddings()
    
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        print("ChromaDB vector store not found. Initializing indexing pipeline...")
        pdf_path = "./docs/rules.pdf"
        if not os.path.exists(pdf_path):
            print(f"Error: {pdf_path} does not exist. Please place a document inside the './docs' folder first.")
            sys.exit(1)
        
        # Load, split, and ingest document
        docs = load_document(pdf_path)
        chunks = split_documents(docs)
        collection = build_vector_store(chunks, emb_model)
    else:
        print("Loading existing ChromaDB vector store...")
        collection = get_collection()
    
    # Initialize retriever and LLM
    retriever = ChromaDBRetriever(collection=collection, embeddings_model=emb_model, k=3)
    llm = OllamaLLM(model_name="gemma3:4b")
    
    # Create LangChain RAG Prompt Template
    prompt_template = ChatPromptTemplate.from_template(
        "You are a helpful customer service assistant. Use only the following retrieved pieces of context to answer the question.\n"
        "If you don't know the answer, say that you don't know. Keep your answer professional and concise.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    
    # Build RAG Chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    print("\nRAG Chain compiled successfully.")
    print("You can now ask questions about the company policy / rules. Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            question = input("\nEnter your question: ").strip()
            if not question:
                continue
            if question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            print("Retrieving context and generating response...")
            answer = rag_chain.invoke(question)
            print(f"\nResponse:\n{answer}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
