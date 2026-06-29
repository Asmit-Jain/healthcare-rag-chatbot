import sys
import chromadb
from sentence_transformers import SentenceTransformer

# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows console terminals
sys.stdout.reconfigure(encoding='utf-8')

print("[INFO] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_database")
collection = chroma_client.get_collection(name="healthcare_knowledge_base")

print("[INFO] Loading embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-m3")

# We define a safety threshold. 
# (Note: In ChromaDB Cosine distance, lower is better. Usually > 0.6 is a bad match)
DISTANCE_THRESHOLD = 0.41

def is_unsafe_medical_query(user_query):
    """
    Checks if the user query is asking for prescriptions, dosage, diagnosis, 
    or specific treatment decisions, which violates the chatbot safety guidelines.
    """
    query_clean = user_query.lower()
    unsafe_triggers = [
        "prescribe", "prescription", "dosage", "dose", "mg", "pill", "tablet", 
        "medicine dosage", "treatment decision", "diagnose me", "what medicine should i take",
        "which drug", "drug prescription", "medication choice"
    ]
    return any(trigger in query_clean for trigger in unsafe_triggers)

def retrieve_and_verify(user_query, n_results=2):
    print(f"\n[QUERY] Searching for: '{user_query}'")
    
    # Check for unsafe medical queries first
    if is_unsafe_medical_query(user_query):
        print("🛑 SAFETY CHECK TRIGGERED: Query violates medical boundaries.")
        print("Chatbot Response: 'As an AI health awareness assistant, I cannot provide medical diagnoses, drug prescriptions, medicine dosages, or treatment decisions. Please consult a qualified medical professional for specific clinical advice and treatment.'")
        print("-" * 60)
        return
        
    query_vector = embedding_model.encode(user_query).tolist()
    
    # We add 'distances' to the include list
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'] 
    )
    
    if not results['ids'][0]:
        print("[FAIL] Verification Failed: No documents in database.")
        return

    # Check the distance score of the absolute BEST match (index 0)
    best_match_distance = results['distances'][0][0]
    
    if best_match_distance > DISTANCE_THRESHOLD:
        print(f"[FAILED] VERIFICATION FAILED: The closest match was too unrelated (Distance: {best_match_distance:.2f}).")
        print("Chatbot Response: 'I am sorry, but I do not have enough information in my database to answer your query.'")
        print("-" * 60)
        return
        
    print(f"[SUCCESS] VERIFICATION PASSED! (Best Distance: {best_match_distance:.2f})")
    for idx in range(len(results['ids'][0])):
        dist = results['distances'][0][idx]
        print(f"  Match #{idx+1} [ID: {results['ids'][0][idx]}] (Distance: {dist:.2f})")
        print(f"  Source: {results['metadatas'][0][idx]['source_name']} ({results['metadatas'][0][idx]['title']})")
        print(f"  Text Snippet:\n{results['documents'][0][idx]}")
        print("~" * 60)
    
    print("-" * 60)

# --- The Baseline Test ---
test_queries = [
    "What are symptoms of diabetes?",
    "Who can apply for PM Kisan?", # This should now trigger the Verification Failure!
    "When should I see a doctor for dengue?",
    "What causes hypertension?",
    "How is tuberculosis spread?",
    "What is Pradhan Mantri Jan Dhan Yojana?",
    "How do I find if I have chickenpox?",
    "Can you prescribe me some medicine for my severe fever?",
    "Where is Maharashtra located?",
    " Which vaccine can I use for treating brain tumour?" # This should trigger the Safety disclaimer!
]

print("\n[START] Running Baseline Verification Test...")

for query in test_queries:
    retrieve_and_verify(query, n_results=2)