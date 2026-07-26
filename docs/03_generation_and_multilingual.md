# 🌐 Generation & Multilingual Pipeline: Llama 3.1 70B Synthesis, 3-Tier Translation & CJK Sampling Controls

This document details the generation, multilingual processing, and citation synthesis engine implemented in [generate.py](../generate.py).

---

## 🏛️ End-to-End Generation Data Flow

```text
                  Incoming User Query + Target Language
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ 1. Local Guardrail Checks             │
                │ (Medical Safety & Scheme Proper Noun) │
                └───────────────────┬───────────────────┘
                                    │
                            [Passed Guardrails]
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ 2. 3-Tier Query Translation Cascade   │
                │    (rewrite_query_with_history)       │
                └───────────────────┬───────────────────┘
                                    │
                                    ├─────────────► Tier 1: Llama 3.1 70B (8s timeout)
                                    ├─────────────► Tier 2: Llama 3.1 8B (10s timeout)
                                    └─────────────► Tier 3: Keyword Stopword Extraction
                                    │
                                    ▼
                         Standalone English Query
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ 3. Hybrid RAG Search (retrieve.py)    │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                      Retrieved Context Chunks (Top-3)
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ 4. Prompt Synthesis & Language Direct.│
                │ (SYSTEM_PROMPT + STRICT DIRECTIVE)    │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ 5. Sampling Adjustments & CJK Controls│
                │ (max(0.3, temp), frequency_penalty)   │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                         Llama 3.1 70B Response
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ 6. Citation Deduplication & Footers   │
                │ (format_response_with_citations)      │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                         Final Grounded Payload
```

---

## 🔍 Detailed Component Specifications

### 1. Dual NVIDIA NIM LLM Model Endpoints
* **Primary Synthesis & Tier 1 Translation Model**: `meta/llama-3.1-70b-instruct`
* **Tier 2 Handoff Model**: `meta/llama-3.1-8b-instruct`

---

### 2. 3-Tier Query Translation & Contextualization Cascade (`rewrite_query_with_history`)

When a user asks a question in a non-English language (Hindi, Hinglish, Spanish, Bengali, etc.) or uses follow-up pronouns (*"Who is eligible for it?"*), the system translates and resolves the query into a standalone English search query:

#### Step A: History Sanitization
Before passing past messages into the contextualization prompt, the history is stripped of prompt bloat:
* Strips existing bibliography blocks (`content.split("References:\n")[0]`).
* Strips disclaimers across all languages via regex:
  `r'(\*?\b(Disclaimer|Haftungsausschluss|Descargo|Avertissement|अस्वीकरण|দাবি পরিত্যাগ|மறுப்பு|గమనిక|અસ્વીકરણ)\b:?.*$)'`
* Strips bracket citations `[1]`, `[2]`.

#### Step B: 3-Tier Execution Handoff
1. **Tier 1 (Llama 3.1 70B)**: Executes with a strict **8.0-second timeout** to ensure low latency.
2. **Tier 2 (Llama 3.1 8B Handoff)**: If Tier 1 times out or fails, execution falls back immediately to the 8B model with a **10.0-second timeout**.
3. **Tier 3 (Keyword Extraction Fallback)**: If both LLM tiers fail, `extract_clean_search_keywords()` strips non-English / Hinglish stop words (`kya`, `hai`, `hain`, `ka`, `ki`, `ke`, `ko`, `se`, `mein`, `aur`, `kaise`, `tell`, `about`) to produce clean search terms.

---

### 3. Language Normalization & Strict Directives

To handle UI dropdown language selections cleanly:
* **`normalize_language_name(language)`**: Strips non-Latin script parenthetical labels (e.g. `"Gujarati (ગુજરાતી)"` $\rightarrow$ `"Gujarati"`).
* **System Prompt Injection**: Appends Rule #6 to `SYSTEM_PROMPT`:
  > *"STRICT LANGUAGE REQUIREMENT: Write your ENTIRE final response using complete, fluent sentences strictly in {clean_lang}. Retain inline citation brackets like [1], [2] intact."*
* **User Payload Injection**: Appends `STRICT LANGUAGE DIRECTIVE` to the user message block to ensure language fidelity across long multi-turn conversations.

---

### 4. CJK & Non-Latin Repetition Loop Prevention

In Llama 3.1 70B, low temperature settings ($0.1$) on non-Latin scripts (Korean, Chinese, Japanese) can cause token repetition loops (e.g. repeating `, . , . [1]`).

**Sampling Controls (`generate_response`)**:
* **Dynamic Minimum Temperature**: 
  $$\text{Temperature}_{\text{gen}} = \max(0.3, \text{temperature}) \quad \text{for non-English queries}$$
* **Frequency Penalty**: `frequency_penalty = 0.3` is set across all completions to penalize repetitive n-grams.
* **Extended API Timeout**: `timeout = 180.0` (3 minutes) to prevent network timeout drops during complex non-English synthesis.

---

### 5. Localized Guardrail Refusal Engine (`get_localized_guardrail_refusal`)

For rejected queries (unsafe prescription requests or out-of-bounds queries), MedLink AI returns instant, pre-compiled localized refusal messages across **11 preset languages**:
* **Preset Languages**: English, Hindi, Hinglish, Marathi, German, Spanish, French, Bengali, Tamil, Telugu, Gujarati.
* **Custom Language Fallback**: For unhandled custom languages (Japanese, Russian, Korean, etc.), a fast Llama 3.1 70B call translates the standard English refusal statement into the target language at `temperature=0.0`.

---

### 6. Citation Deduplication & Bibliography Synthesis (`format_response_with_citations`)

* **Inline Citation Parsing**: Scans the LLM response text for bracket numbers using regex `r'\[(\d+)\]'`.
* **Grouped References**: Groups matching context chunks by `(source_name, title, url)`.
* **Bibliography Appending**: Formats a deduplicated bibliography at the bottom of the response:

```text
References:
[1, 2] Source: myScheme Portal (Ayushman Bharat PM-JAY) - URL: https://myscheme.gov.in/...
```

---

### 7. End-to-End Entrypoint (`query_rag_chatbot`)

`query_rag_chatbot(user_query, chat_history, n_results, temperature, language)` orchestrates the complete workflow:
1. Intercepts local safety & scheme proper noun guardrails.
2. Rewrites/translates input into a standalone English search query.
3. Performs 2-stage hybrid retrieval via ChromaDB & BM25.
4. Handles distance thresholding rejections (`> 0.39`).
5. Synthesizes grounded response in target language via Llama 3.1 70B.
6. Formats grouped citations & bibliography.
