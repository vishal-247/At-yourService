from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    
    print(f"Splitting {len(documents)} document pages into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    from loadDoc import load_document
    docs = load_document("./docs/rules.pdf")
    chunks = split_documents(docs)
    if chunks:
        print(f"First chunk sample:\n{chunks[0].page_content[:200]}...")
