# Healthcare & Government Scheme RAG Chatbot

## 🚀 Overview
This repository contains a high-performance Retrieval-Augmented Generation (RAG) pipeline designed to provide accurate, evidence-based answers to healthcare queries and government scheme eligibility questions. The system leverages WHO Fact Sheets and official Indian government scheme data to minimize hallucinations and ensure strict grounding.

## 🏗️ Pipeline Architecture
Our pipeline follows a **Retrieve-Read-Generate** flow:
1. **Data Ingestion:** Raw JSON data is cleaned, structured, and chunked into 1200-character segments with 200-character overlap.
2. **Hybrid Retrieval:** Uses **BGE-M3 (Dense Vector Search)** for semantic understanding and **BM25 (Sparse Keyword Search)** for exact term matching.
3. **Intent Routing:** An automated router analyzes queries to filter searches by metadata (`category` or `doc_id`), eliminating cross-domain information bleeding.
4. **Safety Guardrails:** A semantic distance verification layer prevents the system from answering out-of-bounds or irrelevant queries.

## 📂 Project Structure
* `chunking.ipynb`: Logic for data cleaning, recursive character splitting, and metadata assignment.
* `embedding-generation.ipynb`: Generates BGE-M3 embeddings using GPU acceleration.
* `build_database.py`: Performs batch ingestion of chunks and vectors into ChromaDB.
* `hybrid_search.ipynb`: Implements the core hybrid search engine with Reciprocal Rank Fusion (RRF).
* `test_retrieval.py`: Script for testing pipeline accuracy and verification guardrails.
* `Data_cleaning_pipeline.ipynb`: Pre-processing tools for scrubbing WHO/Govt JSONs.
* `golden_test_set.json`: Benchmark dataset for evaluating retrieval accuracy.

## 📊 Evaluation Metrics (Phase 2 Baseline)
* **Chunk Recall@5 Rate**: 75.00%
* **Strict Accuracy**: 69.23%
* **Guardrail Safety**: 33.33% (Current rejection rate for out-of-bounds queries)

## ⚖️ Data Sources
* **Medical Data:** Extracted directly from [WHO Fact Sheets](https://www.who.int/news-room/fact-sheets).
* **Government Data:** Sourced from [MyScheme (Government of India)](https://www.myscheme.gov.in/).

## 🛠️ Setup Instructions
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the master JSON files from [Google Drive Link].
4. Run the chunking and embedding notebooks to prepare your local `chroma_database/`.