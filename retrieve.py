import sys
import re
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer

# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows console terminals
sys.stdout.reconfigure(encoding='utf-8')

# --- STEP 1: LOCAL VECTOR DATABASE CONNECTION ---
print("⏳ Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_database")
collection = chroma_client.get_collection(name="healthcare_knowledge_base")

# --- STEP 2: SEMANTIC ENCODING TRANSFORMS CORE INITIALIZATION ---
print("⏳ Loading BGE-M3 embedding model...")
embedding_model = SentenceTransformer("BAAI/bge-m3")

# --- STEP 3: LEXICAL COMPRESSION SETUP (PORTER STEMMER) ---
print("⏳ Initializing Porter Stemmer...")
stemmer = PorterStemmer()

def tokenize(text):
    """A fast tokenizer that removes punctuation, lowercases, and stems words for BM25 matching."""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return [stemmer.stem(w) for w in words]

# --- STEP 4: PAGINATED DATABASE EXTRACTION & LEXICAL CORPUS GENERATION ---
print("⏳ Building the BM25 Keyword Index & extracting scheme vocabulary...")
all_ids = []
tokenized_corpus = []
id_to_category = {}
valid_title_proper_nouns = set()

limit = 2000
offset = 0
all_metadatas = []

while True:
    batch = collection.get(limit=limit, offset=offset, include=['documents', 'metadatas'])
    batch_ids = batch['ids']
    if not batch_ids:
        break
        
    batch_docs = batch['documents']
    batch_metas = batch['metadatas']
    
    all_ids.extend(batch_ids)
    all_metadatas.extend(batch_metas)
    
    for i, doc_id in enumerate(batch_ids):
        tokenized_corpus.append(tokenize(batch_docs[i]))
        meta = batch_metas[i]
        category = meta.get('category', 'General')
        id_to_category[doc_id] = category
        
    offset += limit

# --- STEP 5: BM25 INDEX & PROPER NOUN DICTIONARY MINING ---
bm25_engine = BM25Okapi(tokenized_corpus)

IDF_THRESHOLD = 4.0
for meta in all_metadatas:
    if meta.get('category') == "Government Healthcare Scheme":
        title = meta.get('title', '')
        topic = meta.get('topic', '')
        doc_id = meta.get('doc_id', '')
        source_url = meta.get('source_url', '')
        
        # Title Case proper nouns extraction
        for w in re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', title + " " + topic):
            stem = stemmer.stem(w.lower())
            idf = bm25_engine.idf.get(stem, 9.0)
            if idf > IDF_THRESHOLD:
                valid_title_proper_nouns.add(stem)
                
        # slug / url acronyms extraction
        for text_to_parse in [doc_id, source_url]:
            if text_to_parse:
                last_part = re.split(r'[_/]', text_to_parse)[-1]
                last_part_clean = re.sub(r'\d+$', '', last_part.lower())
                if last_part_clean:
                    stem = stemmer.stem(last_part_clean)
                    idf = bm25_engine.idf.get(stem, 9.0)
                    if idf > IDF_THRESHOLD:
                        valid_title_proper_nouns.add(stem)

# Inject local fallback rules
valid_title_proper_nouns.update([stemmer.stem(x) for x in ['odisha', 'bsky', 'pmjay', 'jssk', 'jsy', 'pmmvy', 'suman', 'goa', 'rajasthan', 'puducherry', 'pondy', 'pondicherry', 'haryana']])
print(f"✅ BM25 Engine successfully indexed {len(tokenized_corpus)} chunks!")
print(f"✅ Extracted {len(valid_title_proper_nouns)} specific proper nouns for scheme verification.")

# --- STEP 6: CORE RETRIEVAL ENGINE ---
DISTANCE_THRESHOLD = 0.39  

def is_unsafe_medical_query(user_query):
    """Checks if the query is asking for prescription, diagnosis, or dosage."""
    query_clean = user_query.lower()
    unsafe_triggers = [
        "prescribe", "prescription", "dosage", "dose", "mg", "pill", "tablet", 
        "medicine dosage", "treatment decision", "diagnose me", "what medicine should i take",
        "which drug", "drug prescription", "medication choice"
    ]
    return any(trigger in query_clean for trigger in unsafe_triggers)

def weighted_reciprocal_rank_fusion(dense_ranks, sparse_ranks, dense_weight=0.85, sparse_weight=0.15, k=60):
    """Combines semantic and keyword rankings using Weighted Reciprocal Rank Fusion."""
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_ranks):
        if doc_id not in rrf_scores: rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += dense_weight * (1.0 / (rank + k))
        
    for rank, doc_id in enumerate(sparse_ranks):
        if doc_id not in rrf_scores: rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += sparse_weight * (1.0 / (rank + k))
        
    sorted_fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_fused_results]

def is_asking_about_scheme(user_query):
    """Identifies if the user is asking about welfare programs."""
    query_clean = user_query.lower()
    scheme_keywords = [
        "yojana", "scheme", "kisan", "bima", "pension", "scholarship", 
        "portal", "loan", "subsidy", "benefit", "assistance", "fund", 
        "apply", "disbursed"
    ]
    return any(kw in query_clean for kw in scheme_keywords)

def passes_proper_noun_check(user_query):
    """Returns True if the query does not contain unrecognized scheme proper nouns."""
    if is_asking_about_scheme(user_query):
        words = re.findall(r'\b\w+\b', user_query)
        for i, w in enumerate(words):
            if w[0].isupper() and i > 0: 
                stem = stemmer.stem(w.lower())
                stem = re.sub(r'\d+$', '', stem)
                idf = bm25_engine.idf.get(stem, 9.0)
                if idf > 5.5 and len(stem) > 1:
                    if stem not in valid_title_proper_nouns:
                        return False
    return True

def evaluate_retrieve(user_query, n_results=5, run_proper_noun_check=True):
    """Performs hybrid search, guardrail filtering, proper noun checks, and round-robin interleaving."""
    # Proper Noun Guardrail Check
    if run_proper_noun_check and not passes_proper_noun_check(user_query):
        return [], 1.0
                
    def split_query(q):
        splits = re.split(r',\s*and\s+|\s+and\s+(?=does|can|what|how|where|is|are|why|if)', q, flags=re.IGNORECASE)
        splits = [s.strip() for s in splits if s.strip()]
        if len(splits) > 1:
            return splits
        return [q]
        
    sub_queries = split_query(user_query)
    dense_lists = []
    sparse_lists = []
    best_distances = []
    
    for sq in sub_queries:
        query_vector = embedding_model.encode(sq).tolist()
        chroma_results = collection.query(
            query_embeddings=[query_vector],
            n_results=50, 
            include=['distances']
        )
        if chroma_results['ids'] and chroma_results['ids'][0]:
            dense_lists.append(chroma_results['ids'][0])
            best_distances.append(chroma_results['distances'][0][0])
        else:
            best_distances.append(1.0)
            
        tokenized_query = tokenize(sq)
        bm25_scores = bm25_engine.get_scores(tokenized_query)
        scored_docs = list(zip(bm25_scores, all_ids))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        sparse_lists.append([doc_id for score, doc_id in scored_docs][:50])
        
    best_semantic_distance = min(best_distances) if best_distances else 1.0
    if best_semantic_distance > DISTANCE_THRESHOLD:
        return [], best_semantic_distance
        
    def interleave_lists(lists):
        interleaved = []
        max_len = max(len(lst) for lst in lists) if lists else 0
        for i in range(max_len):
            for lst in lists:
                if i < len(lst):
                    interleaved.append(lst[i])
        seen = set()
        return [x for x in interleaved if not (x in seen or seen.add(x))]
        
    dense_ids_interleaved = interleave_lists(dense_lists)
    sparse_ids_interleaved = interleave_lists(sparse_lists)
    
    final_ranked_ids = weighted_reciprocal_rank_fusion(dense_ids_interleaved, sparse_ids_interleaved)
    
    filtered_ranked_ids = []
    doc_counts = {}
    for chunk_id in final_ranked_ids:
        doc_id = chunk_id.split('_chunk_')[0] if '_chunk_' in chunk_id else chunk_id
        current_count = doc_counts.get(doc_id, 0)
        if current_count < 3:
            filtered_ranked_ids.append(chunk_id)
            doc_counts[doc_id] = current_count + 1
            
    return filtered_ranked_ids[:n_results], best_semantic_distance

# --- STEP 7: ENTRYPOINT FOR GENERATION PIPELINE ---
def retrieve_for_generation(user_query, n_results=3, run_proper_noun_check=True):
    """
    Fetches actual document content and metadata from ChromaDB for the generation pipeline,
    respecting both safety disclaimer and out-of-bounds guardrails.
    """
    # Guardrail 1: Unsafe medical query
    if is_unsafe_medical_query(user_query):
        return {
            "status": "REJECTED_UNSAFE",
            "message": "As an AI health awareness assistant, I cannot provide medical diagnoses, drug prescriptions, medicine dosages, or treatment decisions. Please consult a qualified medical professional for specific clinical advice and treatment.",
            "chunks": [],
            "distance": 1.0
        }
        
    # Run retrieval
    retrieved_ids, best_distance = evaluate_retrieve(user_query, n_results=n_results, run_proper_noun_check=run_proper_noun_check)
    
    # Guardrail 2: Out-of-bounds / No matched context
    if not retrieved_ids:
        return {
            "status": "REJECTED_OUT_OF_BOUNDS",
            "message": "I am sorry, but I do not have enough information in my database to answer your query.",
            "chunks": [],
            "distance": best_distance
        }
        
    # Retrieve raw text and metadata
    chunks = []
    for doc_id in retrieved_ids:
        doc_data = collection.get(ids=[doc_id], include=['documents', 'metadatas'])
        if doc_data and doc_data['documents']:
            chunks.append({
                "id": doc_id,
                "text": doc_data['documents'][0],
                "metadata": doc_data['metadatas'][0]
            })
            
    return {
        "status": "SUCCESS",
        "chunks": chunks,
        "distance": best_distance
    }
