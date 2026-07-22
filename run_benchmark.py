import sys
# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows console terminals
sys.stdout.reconfigure(encoding='utf-8')

import json
from retrieve import evaluate_retrieve, collection

def run_pipeline_benchmark(test_set_path="golden_test_set.json"):
    print("🚀 Initiating Automated RAG Pipeline Benchmark Evaluation...\n" + "="*85)
    
    # --- STEP 1: LOAD TARGET GOLDEN BENCHMARK DATASET ---
    try:
        with open(test_set_path, 'r', encoding='utf-8') as f:
            evaluation_queries = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{test_set_path}' in your directory. Please save the JSON file first.")
        return
        
    # --- STEP 2: INITIALIZE PERFORMANCE COUNTERS ---
    total_queries = len(evaluation_queries)
    successful_recalls = 0
    strict_matches = 0
    false_positives = 0
    true_negatives = 0
    mrr_sum = 0.0
    hit_at_1 = 0
    fuzzy_matches = 0
    
    print(f"Loaded {total_queries} evaluation targets. Processing query evaluations:\n" + "-"*85)
    
    # --- STEP 3: EVALUATE EACH QUERY THROUGH THE RETRIEVAL ENGINE ---
    for idx, item in enumerate(evaluation_queries):
        query = item['query']
        expected_ids = item['expected_chunk_ids']
        expected_cat = item['expected_category']
        
        # Execute the full hybrid search and guardrail pipeline to retrieve top 5 results
        retrieved_ids, winning_distance = evaluate_retrieve(query, n_results=5)
        
        print(f"\nTest #{idx+1}: '{query}'")
        print(f"  Category: {expected_cat} | Expected Chunks: {expected_ids}")
        
        # --- SCENARIO A: SAFETY GUARDRAIL EVALUATION (OUT-OF-BOUNDS QUERIES) ---
        if len(expected_ids) == 0:
            if len(retrieved_ids) == 0:
                true_negatives += 1  # Guardrail worked perfectly
                status = "✅ PASSED (Blocked cleanly)"
            else:
                false_positives += 1  # Irrelevant data leaked through
                status = f"❌ FAILED (Guardrail Leak: retrieved {retrieved_ids[0]})"
            print(f"  Status: {status} | Distance Score: {winning_distance:.3f}")
            continue  # Move immediately to the next query
            
        # --- SCENARIO B: VALID CONTENT EVALUATION (RANK & ACCURACY TRACKING) ---
        # Find the rank position of the very first matching chunk in the retrieved list
        first_match_rank = -1
        for rank, rid in enumerate(retrieved_ids):
            if rid in expected_ids:
                first_match_rank = rank
                break
        
        # Calculate Mean Reciprocal Rank (MRR) score contribution
        mrr = 1.0 / (first_match_rank + 1) if first_match_rank != -1 else 0.0
        mrr_sum += mrr
        
        # Calculate Top-1 Hit Rate contribution
        if first_match_rank == 0:
            hit_at_1 += 1
            
        # Calculate Chunk-Level intersection matches
        matched_elements = set(expected_ids).intersection(set(retrieved_ids))
        recall_score = len(matched_elements) / len(expected_ids) if len(expected_ids) > 0 else 0.0
        
        if recall_score > 0:
            successful_recalls += 1  # Pipeline captured at least one true target chunk
            
        if recall_score == 1.0:
            strict_matches += 1  # Pipeline successfully captured 100% of required context
            
        # --- OPTIONAL: CALCULATE DOCUMENT-LEVEL FUZZY MATCHES ---
        # Extract document prefixes by dropping individual '_chunk_' identifiers
        expected_docs = {eid.split('_chunk_')[0] for eid in expected_ids if '_chunk_' in eid}
        retrieved_docs = {rid.split('_chunk_')[0] for rid in retrieved_ids if '_chunk_' in rid}
        doc_matched = len(expected_docs.intersection(retrieved_docs)) > 0
        if doc_matched:
            fuzzy_matches += 1
            
        # --- STEP 4: PRINT INDIVIDUAL QUERY LOGGING REPORTS ---
        if recall_score == 1.0:
            status = "✅ FULL HIT (100% chunks retrieved)"
        elif recall_score > 0.0:
            status = f"⚠️ PARTIAL HIT ({len(matched_elements)}/{len(expected_ids)} chunks)"
        elif doc_matched:
            status = "ℹ️ DOCUMENT MATCH (Wrong chunk, correct document)"
        else:
            status = "❌ MISS"
            
        print(f"  Status: {status} | Distance Score: {winning_distance:.3f}")
        print(f"  Retrieved IDs: {retrieved_ids}")
        
        # Fetch and print small snippets from ChromaDB for rapid manual tracking
        for rid in retrieved_ids[:2]:
            doc_data = collection.get(ids=[rid], include=['documents', 'metadatas'])
            if doc_data and doc_data['documents']:
                text = doc_data['documents'][0].strip().replace('\n', ' ')
                snippet = text[:100] + '...' if len(text) > 100 else text
                source = doc_data['metadatas'][0].get('title', 'Unknown Source')
                print(f"    - [{rid}] (Source: {source}): {snippet}")

    # --- STEP 5: CALCULATE AND AGGREGATE FINAL SYSTEM METRICS ---
    content_queries_count = total_queries - (true_negatives + false_positives)
    guardrail_queries_count = true_negatives + false_positives
    
    # Process ratio splits into percentage rates safely
    recall_rate = (successful_recalls / content_queries_count) * 100 if content_queries_count > 0 else 0
    strict_accuracy = (strict_matches / content_queries_count) * 100 if content_queries_count > 0 else 0
    guardrail_accuracy = (true_negatives / guardrail_queries_count) * 100 if guardrail_queries_count > 0 else 0
    mrr_rate = (mrr_sum / content_queries_count) * 100 if content_queries_count > 0 else 0
    hit_at_1_rate = (hit_at_1 / content_queries_count) * 100 if content_queries_count > 0 else 0
    document_match_rate = (fuzzy_matches / content_queries_count) * 100 if content_queries_count > 0 else 0
    
    # Render final metric visualization panel
    print("\n" + "="*85)
    print("📊 FINAL RETRIEVAL PIPELINE METRIC REPORT:")
    print("="*85)
    print(f"🔹 Strict Chunk Recall@5 Rate : {recall_rate:.2f}%")
    print(f"🔹 Strict Multi-Chunk Accuracy: {strict_accuracy:.2f}%")
    print(f"🔹 Document Recall Rate (Fuzzy): {document_match_rate:.2f}%")
    print(f"🔹 Mean Reciprocal Rank (MRR) : {mrr_rate:.2f}%")
    print(f"🔹 Top-1 Hit Rate (Recall@1)  : {hit_at_1_rate:.2f}%")
    print(f"🔹 Guardrail Rejection Safety : {guardrail_accuracy:.2f}%")
    print("="*85)

if __name__ == "__main__":
    run_pipeline_benchmark()
