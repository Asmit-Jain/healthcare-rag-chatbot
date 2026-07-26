# 📊 Data Acquisition, Cleaning & Ingestion Architecture

This document details the reverse-engineered API ingestion, web scraping, text cleaning, recursive chunking, and vector embedding pipelines implemented across [govt_data_extraction.ipynb](../govt_data_extraction.ipynb), [who_factsheets_data_extraction.ipynb](../who_factsheets_data_extraction.ipynb), [Data_Cleaning_Pipeline.ipynb](../Data_Cleaning_Pipeline.ipynb), [chunking.ipynb](../chunking.ipynb), and [embedding-generation.ipynb](../embedding-generation.ipynb).

---

## 🏛️ End-to-End Data Pipeline Architecture

```text
  myScheme.gov.in (288 Schemes)             WHO Factsheets (239 Factsheets)
                │                                         │
                ▼                                         ▼
   [Reverse-Engineered REST API]             [HTML Web Scraping & DOM Traversal]
   govt_data_extraction.ipynb                who_factsheets_data_extraction.ipynb
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

## 🔍 Detailed Data Acquisition & Ingestion Specifications

### 1. Government Schemes Data Acquisition (`govt_data_extraction.ipynb`)

Instead of standard HTML web scraping, data from official Indian Government healthcare portals (`myScheme.gov.in`) was acquired via **Reverse-Engineered Internal REST API Endpoints** identified using Developer Tools:

* **API Endpoints Queried**:
  * `getSchemeDetails`: Fetches scheme metadata, overview, detailed description (`detailedDescription_md`), benefits (`benefits_md`), eligibility criteria (`eligibilityDescription_md`), and exclusions (`exclusions_md`).
  * `getSchemeFaqs`: Fetches official scheme FAQs (question and answer pairs).
  * `getSchemeDocuments`: Fetches required document guidelines.
* **Recursive JSON Parser**:
  Implemented `extract_text_from_rich_json()` to recursively parse deeply nested JSON payload nodes and markdown structures into clean text blocks.
* **Deep Clean & HTML Unescaping**:
  * `deep_unescape(text)`: Recursively decodes heavily nested HTML entities (`&amp;amp;quot;`, `&#39;`, `&nbsp;`).
  * `fix_squished_markdown(text)`: Corrects squished table headers (`****` $\rightarrow$ `** **`).
* **Output Dataset**: Saves **288 verified scheme documents** to `govt_structured_master.json` (2.06 MB).

---

### 2. WHO Disease Factsheets Data Acquisition (`who_factsheets_data_extraction.ipynb`)

Public health awareness factsheets were extracted from the World Health Organization (WHO) website (`www.who.int`) using **HTML Web Scraping & DOM Traversal**:

* **URL Index Extraction**: Scraped the index of WHO factsheet URLs matching `/fact-sheets/detail/` and saved to `who_disease_links.csv`.
* **DOM Traversal & Section Parsing**:
  Used `BeautifulSoup` with `lxml` parser to extract `<h1>` disease titles and sibling `<h2>` section headings. Explicitly preserved list items (`<ul>`, `<ol>`) formatted with bullet prefixes (`- list item`).
* **Noise & Reference Filtering**: Automatically skipped irrelevant non-content sections (`WHO response`, `References`, `Database`, `Related health topics`, `Further reading`).
* **Citation Cleanup**: Removed WHO web citation numbers `(1)`, `(2,3)`.
* **Output Dataset**: Saves **239 verified WHO disease factsheets** to `who_structured_master_cleaned_safe.json` (2.32 MB).

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
