# 🔒 Safety, Authentication & Persistence Architecture

This document details the security guardrails, Bcrypt/JWT authentication engine, and MongoDB Atlas cloud storage implemented in [auth.py](../auth.py), [auth_ui.py](../auth_ui.py), and [db.py](../db.py).

---

## 🏛️ End-to-End Security & Persistence Data Flow

```text
                        Incoming User Request
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │ 1. JWT Authentication Check         │
               │    (verify_access_token)            │
               └──────────────────┬──────────────────┘
                                  │
                          [Valid User Token]
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │ 2. Dual Safety Guardrail Layer      │
               │  - Medical Prescription Intercept   │
               │  - Scheme Proper Noun Verification   │
               └──────────────────┬──────────────────┘
                                  │
                          [Passes Guardrails]
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │ 3. Hybrid RAG & LLM Synthesis       │
               │ (retrieve.py & generate.py)         │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │ 4. MongoDB Atlas Cloud Vault        │
               │  - Save Message Payload             │
               │  - Retain Per-Message Language      │
               └─────────────────────────────────────┘
```

---

## 🔍 Detailed Component Specifications

### 1. Dual-Layer AI Safety Guardrail System

MedLink AI is engineered with programmatically enforced safety boundaries:

#### A. Clinical Safety Intercept (`is_unsafe_medical_query`)
* **Scope**: Blocks clinical diagnostic requests, drug prescriptions, medicine dosages (e.g. mg/pill questions), or treatment decisions.
* **Keywords Monitored**: `prescribe`, `prescription`, `dosage`, `dose`, `mg`, `pill`, `tablet`, `medicine dosage`, `treatment decision`, `diagnose me`, `which drug`, `treatment decision`.
* **Action**: Intercepts the query before database retrieval and returns a localized clinical disclaimer advising the user to consult a certified physician.

#### B. Scheme Proper Noun Verification (`passes_proper_noun_check`)
* **Scope**: Prevents hallucination of fake or out-of-scope government welfare schemes.
* **Mechanism**: Extracts scheme proper nouns using BM25 Inverse Document Frequency filtering (`IDF > 5.5`).
* **Action**: If an unverified scheme noun is detected in a scheme query, the engine returns an immediate localized refusal message.

---

### 2. User Authentication Engine (`auth.py` & `auth_ui.py`)

#### A. Password Security (Bcrypt)
* **Algorithm**: `bcrypt` password hashing with random salt generation (`bcrypt.gensalt(rounds=12)`).
* **Security Guarantee**: Raw passwords are never stored in memory or in the database.

#### B. Authorization Tokens (JSON Web Tokens)
* **Signing Algorithm**: `HS256`.
* **Token Claims Payload**:
  ```json
  {
    "sub": "user@example.com",
    "email": "user@example.com",
    "full_name": "User Name",
    "exp": "2026-08-02T00:00:00Z",
    "iat": "2026-07-26T00:00:00Z"
  }
  ```
* **Expiration**: Tokens expire after **7 days** (`expires_days=7`).
* **Zero-Config Fallback Secret**: Utilizes `JWT_SECRET` environment variable, falling back to an internal default key if omitted.

---

### 3. Cloud Storage & Session Vault (`db.py`)

MedLink AI uses **MongoDB Atlas Serverless Cloud** (`DB_NAME = "health_awareness_rag"`) to store user accounts and multi-turn chat sessions.

#### A. MongoDB Collections
1. **`users` Collection**:
   Stores encrypted user profiles:
   ```json
   {
     "email": "user@example.com",
     "full_name": "User Name",
     "password_hash": "$2b$12$e...",
     "created_at": "2026-07-26T00:00:00Z"
   }
   ```
2. **`chat_sessions` Collection**:
   Stores user chat sessions and historical message threads:
   ```json
   {
     "session_id": "session_123456",
     "user_id": "user@example.com",
     "title": "Ayushman Bharat PM-JAY Query",
     "language": "Hindi (हिंदी)",
     "is_pinned": false,
     "created_at": "2026-07-26T00:00:00Z",
     "updated_at": "2026-07-26T00:05:00Z",
     "messages": [
       {
         "role": "user",
         "content": "Ayushman Bharat kya hai?",
         "timestamp": "2026-07-26T00:00:01Z"
       },
       {
         "role": "assistant",
         "content": "आयुष्मान भारत योजना...",
         "chunks": [...],
         "distance": 0.24,
         "language": "Hindi (हिंदी)",
         "timestamp": "2026-07-26T00:00:03Z"
       }
     ]
   }
   ```

#### B. Message-Level Language Vault
* Each assistant message turn stores its target language (`"language": language`).
* **Benefit**: Ensures historical disclaimers **permanently stay in their original language** when reloading old conversation threads, even if the UI language selector changes later.

#### C. Full CRUD Helper Functions
* **Session Management**: `create_chat_session()`, `get_all_sessions()`, `delete_session()`, `update_session_title()`, `toggle_pin_session()`.
* **Message Append**: `append_message_to_session()` with metadata logging.
