import json
import chromadb

print("⏳ Initializing Local ChromaDB...")

# 1. Initialize a persistent local database
# This creates a folder named "chroma_database" on your computer to store the data
chroma_client = chromadb.PersistentClient(path="./chroma_database")

# 2. Create a unified collection
# We use "cosine" similarity, which is the mathematical standard for BGE-M3 embeddings
collection = chroma_client.get_or_create_collection(
    name="healthcare_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

def ingest_to_chroma(filepath):
    print(f"\n📖 Reading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # ChromaDB requires us to separate our JSON keys into distinct lists
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for item in data:
        ids.append(item["chunk_id"])
        documents.append(item["text"])
        metadatas.append(item["metadata"])
        embeddings.append(item["embedding"])

    # 3. Add data in batches
    # Databases prefer getting data in chunks (e.g., 1000 items at a time) to prevent RAM crashes
    batch_size = 1000
    for i in range(0, len(ids), batch_size):
        print(f"   ➔ Adding batch {i} to {i + len(ids[i:i+batch_size])}...")
        
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size]
        )

print("--- Starting Database Ingestion ---")

# Pass your downloaded files here
ingest_to_chroma("govt_chunks_embedded.jsonl")
ingest_to_chroma("who_chunks_embedded.jsonl")

# Final sanity check
print(f"\n✅ Success! Total chunks safely stored in ChromaDB: {collection.count()}")