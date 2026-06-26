import ollama
from text_splitter import texts

response = ollama.embed(
    model='qwen3-embedding:0.6b',
    input=texts,
)
print(response.embeddings)