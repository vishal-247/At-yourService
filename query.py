from .chromadb import collection

question=input("Enter your question: ")

results = collection.query(
    query_texts=[question], 
    n_results=2 # how many results to return
)

print(results)


# Querying local llm 
import ollama

# Query the local Gemma model
response = ollama.chat(
    model='gemma3:4b',
    messages=[
        {
            'role': 'You are a helpful customer service assistant. Use only the following retrieved pieces of context to answer the question. If you don\'t know the answer, say you don\'t know.',
            'content': results,
            'question': question
        },
    ]
)


print(response['message']['content'])


