# 🏥 MedLink AI — Production-Grade Clinical Awareness & Multilingual Health Intelligence

<p align="center">
  <img src="medlink_logo.png" alt="MedLink AI Logo" width="120">
</p>

<p align="center">
  <strong>Grounded Multilingual RAG Assistant for Public Healthcare Education & Government Scheme Navigation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.33%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/ChromaDB-VectorDB-000000?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Embeddings-BAAI%2Fbge--m3-blueviolet?style=for-the-badge" alt="BGE-M3">
  <img src="https://img.shields.io/badge/LLM-Llama_3.1_70B-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA NIM">
  <img src="https://img.shields.io/badge/MongoDB-Atlas_Vault-47A248?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB">
  <img src="https://img.shields.io/badge/Guardrails-100%25_Verified-success?style=for-the-badge" alt="Guardrail Safety">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

---

## 📌 Executive Summary

**MedLink AI** is an advanced, production-grade Generative AI Healthcare Assistant built to provide reliable, source-grounded public health education and government healthcare scheme navigation across India. Powered by a **2-Stage Hybrid Dense (BGE-M3) + Lexical (BM25) Reciprocal Rank Fusion (RRF)** retrieval pipeline and **Llama 3.1 70B**, MedLink AI synthesizes accurate, factual responses strictly from official government health portals (`myScheme.gov.in`) and World Health Organization (WHO) factsheets.

> [!IMPORTANT]  
> **Strict Non-Diagnostic Mission Statement**:  
> MedLink AI is designed strictly for **preventive care, wellness awareness, and government scheme education**. It does **NOT** provide medical diagnoses, drug prescriptions, medicine dosages, or clinical treatment decisions. Any unsafe diagnostic query is programmatically intercepted by strict safety guardrails that advise consulting a qualified medical professional.

---

## 📈 Empirical Evaluation Benchmark Results

The retrieval and safety architecture of MedLink AI was empirically benchmarked against a **50-Query Golden Evaluation Suite** (`run_benchmark.py`). Below are the quantitative performance results demonstrating high precision, high recall, and strict safety enforcement:

| Benchmark Parameter | Metric Score | Clinical & Technical Significance |
|:---|:---:|:---|
| 🎯 **Strict Chunk Recall@5** | **90.00%** | **9 out of 10 queries** retrieve the exact target chunk within the top 5 results. |
| 🎯 **Strict Multi-Chunk Accuracy** | **55.00%** | Exact multi-part chunk retrieval accuracy for multi-concept health queries. |
| 📚 **Document Recall Rate (Fuzzy)** | **100.00%** | **100% success rate** in identifying and retrieving the correct source document. |
| ⚡ **Mean Reciprocal Rank (MRR)** | **74.25%** | High ranking density; target chunks appear predominantly at Top-1 or Top-2. |
| 🥇 **Top-1 Hit Rate (Recall@1)** | **62.50%** | Nearly **2/3 of queries** place the perfect answer chunk as the very 1st result. |
| 🛡️ **Guardrail Rejection Safety** | **100.00%** | **100% rejection accuracy** for off-topic, prescription, and unsafe medical queries. |

---

## 🏛️ End-to-End System Architecture

MedLink AI implements a modular, high-throughput RAG architecture engineered for low retrieval latency, multi-turn history resolution, and zero-hallucination grounded response synthesis.

```text
+-----------------------------------------------------------------------------------+
|                                 USER QUERY INPUT                                  |
|                (English, Hindi, Hinglish, Spanish, Bengali, etc.)                 |
+-----------------------------------------┬-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            MEDICAL SAFETY & GUARDRAILS                            |
|             Checks: Unsafe Prescriptions / Dosages & Scheme Proper Nouns          |
+-----------------------------------------┬-----------------------------------------+
                                          |
                              [Passes Safety Guardrails]
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        3-TIER QUERY TRANSLATION CASCADE                           |
|        Tier 1: Llama 3.1 70B  -->  Tier 2: Llama 3.1 8B  -->  Tier 3: BM25        |
|        (Resolves multi-turn history & translates query to English search)         |
+-----------------------------------------┬-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            2-STAGE HYBRID RRF SEARCH                              |
|  +----------------------------------------┬------------------------------------+  |
|  |         Dense Semantic Search          |       Sparse Lexical Search        |  |
|  |       (BAAI/bge-m3 + ChromaDB)         |    (BM25Okapi + Porter Stemmer)    |  |
|  +----------------────┬───────────────────+──────────────────┬─────────────────+  |
|                       │                                      │                    |
|                       └───────────────────┬──────────────────┘                    |
|                                           │                                       |
|                              Weighted RRF (0.85 / 0.15)                           |
+-----------------------------------------┬-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           DISTANCE THRESHOLD & DIVERSITY                          |
|             - 0.39 Distance Cutoff Filter (Rejects out-of-bounds queries)         |
|             - Max 3 Chunks / Document (Prevents source monopolization)            |
+-----------------------------------------┬-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         LLM RESPONSE GENERATION & SYNTHESIS                       |
|         Llama 3.1 70B Instruct (NVIDIA NIM) + Inline Citations [1], [2]           |
+-----------------------------------------┬-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           PERSISTENT SESSION & UI VAULT                           |
|        MongoDB Atlas Vault + Message-Specific Language Disclaimers & RAG UI       |
+-----------------------------------------------------------------------------------+
```

---

## 📊 Healthcare Corpus & Data Provenance

MedLink AI's knowledge base indexes **8,461 verified text chunks** extracted exclusively from official, public healthcare sources. The dataset covers both macro-level public health schemes across India and disease-specific awareness factsheets from the World Health Organization.

| Source Domain | Document Category | Information Scope | Primary Target Schemes / Topics |
|:---|:---|:---|:---|
| 🏛️ **`myScheme.gov.in`** | Government Healthcare Schemes | Welfare guidelines, financial assistance rules, eligibility criteria, application portals | Ayushman Bharat PM-JAY, PMMVY, SUMAN, JSSK, JSY, BSKY (Odisha), Goa Mediclaim, Chiranjeevi (Rajasthan), etc. |
| 🌐 **World Health Organization (WHO)** | Disease & Health Factsheets | Clinical symptoms, preventive care, transmission modes, vaccination schedules | Malaria, Tuberculosis, Hypertension, Diabetes, Dengue, Sepsis, Stroke, Asthma, HIV/AIDS, Hepatitis, etc. |

---

## ⚙️ Hybrid Search & Retrieval (RAG)

To get accurate answers quickly, MedLink AI combines two search methods into a simple **2-stage retrieval pipeline**:

* **Semantic Search (BGE-M3)**: Understands the meaning of the query (for example, connecting *"high blood pressure"* to *Hypertension*).
* **Keyword Search (BM25)**: Matches exact words, scheme names (like `PM-JAY` or `PMMVY`), and numerical values.
* **Score Fusion (RRF)**: Combines the top 50 results from both searches using Reciprocal Rank Fusion (85% semantic weight + 15% keyword weight).
* **Safety Cutoff Filter (0.39 Threshold)**: Rejects questions that have a distance score greater than `0.39`, preventing the AI from making up answers when information is missing.
* **Source Diversity Cap**: Limits results to a maximum of **3 chunks per document** so answers draw from multiple trusted sources.

---

## 🌐 Multilingual Support & Chat Memory

MedLink AI answers questions in **11+ preset languages** (English, Hindi, Hinglish, Bengali, Tamil, Telugu, Marathi, Gujarati, Spanish, French, German) as well as any custom language (Japanese, Korean, Punjabi, etc.).

* **Follow-up Query Understanding**: Automatically rewrites follow-up questions (like *"Who is eligible for it?"*) into complete search queries using recent conversation context.
* **Query Translation**: Translates questions typed in regional or foreign languages into English search terms to retrieve facts from the database.
* **Clean Text Generation**: Automatically adjusts temperature settings and adds repetition penalties (`frequency_penalty=0.3`) for languages like Korean, Chinese, or Japanese to ensure fluent, natural responses.

---

## 🎨 User Interface & RAG Inspector Diagnostics

MedLink AI features a modern, dark-mode glassmorphic user interface built with Streamlit and custom CSS styling.

* **🔍 RAG Inspector (Real-Time Diagnostics)**: Click open the `🔍 RAG Inspector` drawer below any assistant message to inspect:
  * **Vector Semantic Distance**: View the exact cosine distance score (e.g. `0.24`) compared against the `0.39` cutoff threshold.
  * **Retrieved Chunks & Metadata**: View the exact text snippets, source document titles, categories, doc IDs, and source URLs.
* **Interactive Elements**:
  * **Clickable Citation Chips**: Displays glowing source chips (`[1] WHO Factsheet`, `[2] myScheme Portal`) for easy verification.
  * **Execution Latency Badge**: Displays real-time response generation speed (`⚡ Response Time: 1.42s`).

---

## 🔒 User Accounts & Session Vault (MongoDB Atlas)

MedLink AI includes a full-stack user authentication system and cloud database storage.

* **Secure Authentication**:
  * Passwords are securely encrypted using **Bcrypt** hashing.
  * User sessions are authorized using **JSON Web Tokens (JWT)**.
  * Clean 2-column login and signup portal (`auth_ui.py`).
* **MongoDB Atlas Cloud Persistence**:
  * Stores user accounts, active chat sessions, and message history in the cloud (`db.py`).
  * **Sidebar Session Control**: Users can start new chats, search past conversations, and rename, pin, or delete chat sessions.
* **Message-Level Language Vault**:
  * Saves the target language (`"language": target_lang_turn`) for every message turn.
  * Ensures historical disclaimers **permanently stay in their original language** when reloading old conversations.

---

## 📁 Project Directory Structure

```text
Health_Awareness_Chatbot/
│
├── app.py                  # Streamlit Main App UI & Chat Interface
├── generate.py             # Llama 3.1 70B/8B Translation & Response Generation
├── retrieve.py             # 2-Stage Hybrid RAG (BGE-M3 + BM25 RRF) Search Engine
├── db.py                   # MongoDB Atlas Persistence (Users, Sessions, Messages)
├── auth.py                 # Security Helpers (Bcrypt Password Hashing & JWT Tokens)
├── auth_ui.py              # User Login & Signup Form Components
├── run_benchmark.py        # 50-Query Golden Benchmark Evaluation Suite
│
├── chroma_database/        # Persistent Local Vector Store (8,461 Embeddings)
├── golden_test_set.json    # 50 Ground-Truth Test Cases for Benchmarking
├── requirements.txt        # Python Dependencies & Libraries
├── medlink_logo.png        # MedLink AI Logo Asset
└── README.md               # Project Documentation
```