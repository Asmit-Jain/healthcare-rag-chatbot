# 📊 Data Acquisition, Cleaning & Ingestion Architecture

This document details the web scraping, text cleaning, recursive chunking, and vector embedding pipelines implemented across [govt_data_extraction.ipynb](../govt_data_extraction.ipynb), [who_factsheets_data_extraction.ipynb](../who_factsheets_data_extraction.ipynb), [Data_Cleaning_Pipeline.ipynb](../Data_Cleaning_Pipeline.ipynb), [chunking.ipynb](../chunking.ipynb), and [embedding-generation.ipynb](../embedding-generation.ipynb).

---

## 🏛️ End-to-End Data Pipeline Architecture

```text
  myScheme.gov.in (288 Schemes)             WHO Factsheets (239 Factsheets)
                │                                         │
                ▼                                         ▼
   govt_data_extraction.ipynb               who_factsheets_data_extraction.ipynb
                │                                         │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                        Data_Cleaning_Pipeline.ipynb
                                     │
                                     ▼
                   Master Cleaned JSON Datasets (~4.3 MB)
                   - govt_structured_master.json
                   - who_structured_master_cleaned_safe.json
                                     │
                                     ▼
                              chunking.ipynb
                                     │
                                     ▼
                       Raw Text Chunk Files (~7.3 MB)
                       - govt_chunks_raw.jsonl
                       - who_chunks_raw.jsonl
                                     │
                                     ▼
                    embedding-generation.ipynb (Kaggle GPU)
                                     │
                                     ▼
                     Embedded Vector Files (8,461 Chunks)
                     - govt_chunks_embedded.jsonl
                     - who_chunks_embedded.jsonl
                                     │
                                     ▼
                             build_database.py
                                     │
                                     ▼
                    ChromaDB Vector Store (./chroma_database)
```

---

## 🔍 Detailed Component Specifications

### 1. Data Acquisition (Web Scraping)
* **Government Schemes (`myScheme.gov.in`)**:
  Extracted 288 official welfare scheme pages across India using BeautifulSoup and Requests. Extracted structured metadata: scheme title, category, eligibility guidelines, financial assistance rules, and official application links.
* **WHO Disease Factsheets**:
  Extracted 239 disease and public health awareness factsheets from World Health Organization portals. Extracted clinical symptoms, transmission modes, preventive measures, and treatment overviews.

---

### 2. Data Cleaning & Structured Master Datasets
* **Cleaning Rules**: Removed HTML boilerplates, navigation headers, disclaimers, duplicate whitespace, and unverified web scripts.
* **Output Artifacts**:
  * `govt_structured_master.json`: 288 master scheme records (2.06 MB).
  * `who_structured_master_cleaned_safe.json`: 239 master WHO factsheet records (2.32 MB).

---

### 3. Recursive Text Chunking & Context Prefixing (`chunking.ipynb`)

Text chunking is performed using LangChain's `RecursiveCharacterTextSplitter` optimized for `BAAI/bge-m3` embedding token limits.

#### A. Chunking Hyperparameters
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    length_function=len,
    separators=[r"\n\n", r"\n", r"(?<=\. )", r" ", r""],
    is_separator_regex=True
)
```

#### B. Text Cleaning & Normalization (`clean_text`)
* Replaces encoding artifacts (`"[?]"` $\rightarrow$ `"₹"`).
* Strips raw web citation numbers `[1]`, `[22]`, `(1)`, `(22)`.
* Strips single hidden line breaks while preserving paragraph boundaries.

#### C. Section-Based Logic & Metadata Enrichment
* **FAQ Sections**: Extracted as complete, un-truncated standalone FAQ chunks (`chunk_type: "faq"`).
* **General Sections**: Split using `RecursiveCharacterTextSplitter` into structured chunks (`chunk_type: "general"`).
* **Contextual Prefixing**: Every chunk's text is prefixed with its document title and section heading:
  ```text
  Document: {metadata['title']}
  Section: {section_title}

  {chunk_text}
  ```
* **Output Chunks**: Produces **8,461 verified text chunks** (`govt_chunks_raw.jsonl` & `who_chunks_raw.jsonl`).

---

### 4. Vector Embedding Generation (`embedding-generation.ipynb`)
* **Embedding Model**: `BAAI/bge-m3` (1024-dimensional dense vector embeddings).
* **GPU Acceleration**: Executed on free Kaggle / Google Colab GPUs (T4/P100), computing all 8,461 vector embeddings in **under 2 minutes**.

---

### 5. Vector Database Ingestion (`build_database.py`)
* Ingests the embedded `.jsonl` files into local ChromaDB (`./chroma_database`) under the `healthcare_knowledge_base` collection configured with **Cosine Distance metric space**.
