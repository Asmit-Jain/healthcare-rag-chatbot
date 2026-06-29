# Implementation Plan - Balanced Golden Test Set (50-Query Version)

This plan details the creation of a new, verified golden test set containing **50 queries** to exhaustively evaluate our RAG retrieval pipeline across all edge cases while keeping the benchmark runtime fast.

## Guidelines to Follow
1. **Balanced Chunks**: 16 pure disease queries (WHO factsheets) and 16 pure scheme queries (Gov schemes).
2. **Out-of-Bounds Check**: 10 queries testing rejection of off-topic categories (general knowledge, agriculture, finance, tech).
3. **Mixed Queries**: 8 queries asking for medical details combined with government financial support (cross-domain searches).
4. **Single vs. Multi-Chunk Split**: Balanced mix of targeted single-chunk questions (FAQs) and broader multi-chunk queries (symptoms, application steps).
5. **Lexical vs. Semantic Variety**: A mix of queries using exact keyword matching and queries using synonyms/paraphrasing.
6. **Minor Typos / Noise (Robustness Check)**: A subset of queries (3-4) containing minor typos (e.g. "diabtes", "yojna") to verify stemmer/embedding robustness.
7. **Disease & Scheme Diversity**: Questions will cover a wide variety of different diseases and schemes across the dataset, rather than concentrating on a few.

---

## Execution Phases (1 Prompt Per Phase)

To guarantee zero mapping errors and avoid token constraints, we split the generation into **4 Prompts**:

### Phase 1 (Prompt 1) — 13 Queries
- 6 Pure Disease queries (covering 6 different diseases: e.g., Bipolar, Blindness, COVID-19, Chagas, Dengue, Diabetes).
- 7 Pure Scheme queries (covering 7 different schemes: e.g., JSSK, JSY, SUMAN, PM POSHAN, Biju Swasthya Kalyan Yojana, BPL/SC-ST specific benefits, State disease funds).

### Phase 2 (Prompt 2) — 13 Queries
- 7 Pure Disease queries (covering 7 different diseases: e.g., Mpox, Tuberculosis, Cancer, Ebola, Anaemia, Influenza, Hepatitis).
- 6 Pure Scheme queries (covering 6 different schemes: e.g., PMMVY, Swachh Bharat Mission, Silicosis support, Chief Minister's Relief Fund, Ayushman Bharat Wellness Centres, Goa Mediclaim).

### Phase 3 (Prompt 3) — 12 Queries
- 8 Mixed Queries combining disease awareness with relevant government schemes (e.g., Cancer symptoms + Oncology schemes, TB spread + Nikshay Poshan Yojana).
- 4 Out-of-Bounds queries (non-healthcare schemes, e.g., PM Kisan, PM Fasal Bima).

### Phase 4 (Prompt 4) — 12 Queries
- 6 Out-of-Bounds queries (general knowledge, unrelated tech, math, other industries).
- Compile and verify all 50 queries, overwrite `golden_test_set.json`, and run `scratch/run_benchmark.py` to finalize.

---

## Proposed Changes

### `golden_test_set.json`

#### [NEW] [golden_test_set.json](file:///c:/Users/ASMIT%20JAIN/Desktop/Summer_2026_Internship/Health_Awareness_Chatbot/golden_test_set.json)
We will overwrite this file in Phase 4.

---

## Verification Plan

We will run the benchmark suite to verify the metrics:
```bash
python scratch/run_benchmark.py
```
