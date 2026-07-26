# ⚙️ Retrieval Architecture: 2-Stage Hybrid Dense & Sparse RRF

This document provides a comprehensive technical reference for the **2-Stage Hybrid Retrieval Engine** implemented in [retrieve.py](../retrieve.py).

---

## 🏛️ End-to-End System Pipeline Flow Diagram

```text
                                 ┌───────────────────────────┐
                                 │    Incoming User Query    │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ 1. Unsafe Medical Check   │
                                 │ (is_unsafe_medical_query) │
                                 └─────────────┬─────────────┘
                                               │
                                       [Safe Query Pass]
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ 2. Scheme Proper Noun Check│
                                 │ (passes_proper_noun_check)│
                                 └─────────────┬─────────────┘
                                               │
                                        [Valid Scheme Pass]
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │    3. Query Splitting     │
                                 │       (split_query)       │
                                 └─────────────┬─────────────┘
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                            Sub-Query 1                 Sub-Query 2
                                 │                           │
                        ┌────────┴────────┐         ┌────────┴────────┐
                        ▼                 ▼         ▼                 ▼
                     BGE-M3            BM25      BGE-M3            BM25
                    Semantic          Lexical   Semantic          Lexical
                     Search            Search    Search            Search
                        │                 │         │                 │
                        └────────┬────────┘         └────────┬────────┘
                                 │                           │
                                 ▼                           ▼
                           Dense Results 1             Dense Results 2
                           Sparse Results 1            Sparse Results 2
                                 │                           │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ 4. Round-Robin Interleave │
                                 │    (interleave_lists)     │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ 5. Weighted RRF Fusion    │
                                 │    (Dense: 0.85, BM25: 0.15)│
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ 6. Distance Cutoff Check  │
                                 │ (DISTANCE_THRESHOLD = 0.39)│
                                 └─────────────┬─────────────┘
                                               │
                                       [Distance <= 0.39]
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ 7. Document Diversity Cap │
                                 │  (Max 3 Chunks / Document)│
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Grounded Context Payload  │
                                 │(status, chunks, distance) │
                                 └───────────────────────────┘
```

---

## 🔍 Detailed Component Specifications

### 1. Database Connection & Corpus Indexing (`Steps 1–5`)
* **ChromaDB Vector Store**: Connects to persistent SQLite database (`./chroma_database`) using collection `healthcare_knowledge_base` configured with **Cosine Similarity metric space** (`metadata={"hnsw:space": "cosine"}`).
* **Dense Embedding Model**: `BAAI/bge-m3` producing 1024-dimensional dense vectors.
* **Paginated BM25 Ingestion**: Queries ChromaDB in `limit=2000` batch pagination loops to extract all documents into RAM and builds a `BM25Okapi` sparse keyword index over Porter-stemmed text tokens (`tokenize(text)` using `nltk.stem.PorterStemmer`).
* **Scheme Vocabulary Mining**: Mines scheme proper nouns from titles and URLs where `IDF > 4.0`, combined with manual regional scheme fallback keywords (`bsky`, `pmjay`, `jssk`, `jsy`, `pmmvy`, `suman`, `odisha`, `goa`, `rajasthan`, `puducherry`, `haryana`).

---

### 2. Safety Guardrails & Verification Logic

#### A. Unsafe Medical Query Interception (`is_unsafe_medical_query`)
Scans queries for diagnostic, drug prescription, or dosage triggers:
* **Trigger Keywords**: `prescribe`, `prescription`, `dosage`, `dose`, `mg`, `pill`, `tablet`, `medicine dosage`, `treatment decision`, `diagnose me`, `which drug`, `drug prescription`.
* **Action**: Immediately halts execution and returns status `REJECTED_UNSAFE` with an official clinical advice disclaimer.

#### B. Scheme Proper Noun Verification (`passes_proper_noun_check`)
Prevents hallucinations for unrecognized or out-of-scope government welfare programs:
* Identifies scheme queries via `is_asking_about_scheme` triggers (`yojana`, `scheme`, `kisan`, `bima`, `pension`, `scholarship`, `portal`, `loan`, `subsidy`, `benefit`, `assistance`).
* Extracts capitalized proper nouns and checks if their stemmed IDF score is $> 5.5$.
* **Action**: If a high-IDF scheme noun is not found in `valid_title_proper_nouns`, retrieval is halted instantly.

---

### 3. Query Decomposition & Round-Robin Interleaving

#### A. Coordinated Query Splitting (`split_query`)
Splits compound questions into standalone sub-queries:
* **Regex Rule**: `re.split(r',\s*and\s+|\s+and\s+(?=does|can|what|how|where|is|are|why|if)', q)`
* **Example**: *"What is Ayushman Bharat **and** what are its benefits?"* $\rightarrow$ `["What is Ayushman Bharat", "what are its benefits"]`

#### B. Round-Robin Interleaving (`interleave_lists`)
For multi-part queries, candidate chunk IDs from all sub-queries are interleaved sequentially without duplicates:
$$L_{\text{interleaved}} = [L_{1}[0], L_{2}[0], L_{1}[1], L_{2}[1], \dots]$$
This guarantees equal representation of facts for every part of a complex question.

---

### 4. Stage 1: Dual Dense & Sparse Retrieval

For every sub-query:
1. **Dense Vector Search**: Encodes sub-query using `bge-m3` and queries ChromaDB for Top-50 nearest neighbors.
2. **Sparse Lexical Search**: Tokenizes sub-query using `PorterStemmer` and queries the `BM25Okapi` index for Top-50 keyword matches.

---

### 5. Stage 2: Weighted Reciprocal Rank Fusion (RRF)

Combines the top-50 dense and sparse candidate lists into a unified relevance score. Dense semantic search is weighted at **85%** and sparse keyword search at **15%** ($k=60$):

$$\text{RRF Score}(d) = 0.85 \cdot \frac{1}{60 + r_{\text{dense}}(d)} + 0.15 \cdot \frac{1}{60 + r_{\text{sparse}}(d)}$$

---

### 6. Post-Retrieval Filtering & Diversity Enforcement

#### A. Out-of-Bounds Distance Threshold (`DISTANCE_THRESHOLD = 0.39`)
* Evaluates the best retrieved chunk's Cosine Distance:
  $$\text{Cosine Distance} = 1 - \text{Cosine Similarity}$$
* **Threshold Rule**: If $\text{Distance}_{\text{best}} > 0.39$, the query is classified as out-of-bounds, returning status `REJECTED_OUT_OF_BOUNDS`.

#### B. Document Diversity Cap (`doc_counts < 3`)
* Prevents a single document from monopolizing the prompt context window.
* **Cap Rule**: Maximum **3 chunks per parent document ID** (`chunk_id.split('_chunk_')[0]`).

---

### 7. Entrypoint Interface (`retrieve_for_generation`)

The production interface returned to `generate.py` outputs a structured payload:

```python
{
    "status": "SUCCESS" | "REJECTED_UNSAFE" | "REJECTED_OUT_OF_BOUNDS",
    "chunks": [
        {
            "id": "pmjay_scheme_chunk_0",
            "text": "Ayushman Bharat PM-JAY provides health coverage of Rs. 5 Lakhs...",
            "metadata": {"title": "PM-JAY", "source_url": "https://myscheme.gov.in/..."}
        }
    ],
    "distance": 0.24
}
```
