import chromadb
from sentence_transformers import SentenceTransformer

print("⏳ Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_database")
collection = chroma_client.get_collection(name="healthcare_knowledge_base")

print("⏳ Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-m3")

# We define a safety threshold. 
# (Note: In ChromaDB Cosine distance, lower is better. Usually > 0.6 is a bad match)
DISTANCE_THRESHOLD = 0.3

def retrieve_and_verify(user_query, n_results=3):
    print(f"\n🔍 Searching for: '{user_query}'")
    
    query_vector = embedding_model.encode(user_query).tolist()
    
    # We add 'distances' to the include list
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'] 
    )
    
    if not results['ids'][0]:
        print("❌ Verification Failed: No documents in database.")
        return

    # Check the distance score of the absolute BEST match (index 0)
    best_match_distance = results['distances'][0][0]
    
    if best_match_distance > DISTANCE_THRESHOLD:
        print(f"🛑 VERIFICATION FAILED: The closest match was too unrelated (Distance: {best_match_distance:.2f}).")
        print("🤖 Chatbot Response: 'I am sorry, but I do not have enough information in my database to answer your query.'")
        print("-" * 60)
        return
        
    print(f"✅ VERIFICATION PASSED! (Best Distance: {best_match_distance:.2f})")
    print(f"📌 Match #1 [ID: {results['ids'][0][0]}]")
    print(f"📄 Source: {results['metadatas'][0][0]['source_name']} ({results['metadatas'][0][0]['title']})")
    print(f"📝 Text Snippet:\n{results['documents'][0][0]}")
    
    print("-" * 60)

# --- The Baseline Test ---
test_queries = [
    "What are symptoms of diabetes?",
    "Who can apply for PM Kisan?", # This should now trigger the Verification Failure!
    "When should I see a doctor for dengue?",
]

print("\n🚀 Running Baseline Verification Test...")

for query in test_queries:
    retrieve_and_verify(query, n_results=1)