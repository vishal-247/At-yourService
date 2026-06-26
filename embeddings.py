import ollama
from langchain_core.embeddings import Embeddings
from typing import List

class OllamaEmbeddings(Embeddings):
    """
    Custom LangChain Embeddings wrapper for Ollama.
    """
    model_name: str = 'qwen3-embedding:0.6b'

    def __init__(self, model_name: str = 'qwen3-embedding:0.6b'):
        super().__init__()
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = ollama.embed(model=self.model_name, input=texts)
        return response.embeddings

    def embed_query(self, text: str) -> List[float]:
        response = ollama.embed(model=self.model_name, input=text)
        return response.embeddings[0]

if __name__ == "__main__":
    # Test embeddings
    print("Testing OllamaEmbeddings...")
    embeddings = OllamaEmbeddings()
    sample_texts = ["Hello world", "RAG chat bot"]
    vectors = embeddings.embed_documents(sample_texts)
    print(f"Generated {len(vectors)} vector(s) of dimension {len(vectors[0])}.")