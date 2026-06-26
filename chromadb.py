import chromadb
from embeddings import response


chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="company_policy")

ids = []
documents = []
embeddings = []
for item in response:
    ids.append(item["id"])
    documents.append(item.get("document", ""))
    embeddings.append(item.get("embedding"))

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
)


