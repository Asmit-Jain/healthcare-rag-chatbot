import sys
# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows console terminals
sys.stdout.reconfigure(encoding='utf-8')

import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import retrieve_for_generation

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client using NVIDIA's API endpoint
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("Error: NVIDIA_API_KEY environment variable is not set in the .env file.")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Llama 3.1 70B Instruct model on NVIDIA NIM
MODEL_NAME = "meta/llama-3.1-70b-instruct"

# --- STEP 4: SYSTEM PROMPT WITH STRICT GROUNDING AND MEDICAL GUARDRAILS ---
SYSTEM_PROMPT = """You are a professional, cautious, and helpful Healthcare Awareness AI Assistant. Your goal is to answer the user's query by strictly using ONLY the facts provided in the "Retrieved Context" section below.

Follow these strict rules at all times:
1. Grounding: Answer the query using ONLY the provided Retrieved Context. If the answer cannot be found or reasonably inferred from the context, state clearly and politely: "I am sorry, but I do not have enough information to answer your query." Do not use any external knowledge or make up facts.
2. Medical Boundaries: You are NOT a doctor or medical professional. 
   - DO NOT diagnose conditions, prescribe medications, or recommend specific drug dosages or treatment decisions under any circumstances.
   - If the user asks for prescriptions, dosages, or diagnostic medical treatment, you must refuse to answer and advise them to consult a qualified physician.
3. Citation Rule: When you state a fact from a retrieved chunk, you MUST cite it using inline brackets matching its chunk number (e.g., [1], [2]).
4. Tone: Maintain a highly cautious, educational, and objective tone. Always append the following medical disclaimer at the very end of any response discussing symptoms or conditions:
   "Disclaimer: This information is for educational purposes only. Please consult a qualified medical professional for specific clinical advice and treatment."
"""

def rewrite_query_with_history(user_query, chat_history, language="English"):
    """
    Uses Llama 3.1 70B to rewrite a follow-up query into a standalone search query if history exists,
    or translates non-English/Hinglish queries into English for accurate ChromaDB & BM25 retrieval.
    """
    if not chat_history and (not language or language == "English"):
        return user_query

    # Assemble conversation history snippet for prompt
    history_str = ""
    if chat_history:
        for msg in chat_history[-10:]:  # Limit to last 10 messages (5 turns) for context
            role = "User" if msg["role"] == "user" else "Assistant"
            # Strip references list from history content to keep it clean
            clean_content = msg["content"].split("References:\n")[0].strip()
            history_str += f"{role}: {clean_content}\n"

    prompt = f"""You are a query translation and contextualization assistant.
Given the following conversation history and a user's question (which may be in any language like Hindi, Hinglish, Spanish, etc.), perform two tasks:
1. Resolve any relative pronouns or references (like 'it', 'this scheme', 'its symptoms') using the conversation history.
2. Translate the question into a clear, standalone, search-friendly query in ENGLISH so it can search an English document vector database.

Do not answer the question. Only return the standalone English search query string.

Conversation History:
{history_str if history_str else "None"}

User Question: {user_query}
Standalone English Search Query:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
            timeout=30.0
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        print(f"🔄 Conversational/Multilingual Context: Rewrote query to English standalone as: '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[WARNING] Query rewriting/translation failed: {e}. Falling back to original query.")
        return user_query

def generate_response(user_query, context_chunks, chat_history=None, temperature=0.1, language="English"):
    """
    Assembles the context prompt, appends chat history, calls the Llama model with target language instructions, and returns response.
    """
    if chat_history is None:
        chat_history = []

    # Format the retrieved context into a single string
    formatted_context = ""
    for idx, chunk in enumerate(context_chunks):
        meta = chunk.get("metadata", {})
        source_name = meta.get("source_name", "Unknown Source")
        doc_title = meta.get("title", "Unknown Title")
        formatted_context += f"---\n[{idx + 1}] Source: {source_name} ({doc_title})\n"
        formatted_context += f"Content: {chunk.get('text', '')}\n"
    formatted_context += "---\n"

    # Dynamic System Prompt with Language Constraint
    sys_prompt = SYSTEM_PROMPT
    if language and language != "English":
        sys_prompt += f"\n5. STRICT LANGUAGE REQUIREMENT: You MUST synthesize and write your entire final response strictly in {language}. Retain inline citation brackets like [1], [2] intact, and ensure any medical disclaimer is written accurately in {language}."

    # Compile messages payload starting with system instructions
    messages = [{"role": "system", "content": sys_prompt}]

    # Append chat history (limit to last 10 messages / 5 turns)
    for msg in chat_history[-10:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Append current turn user query along with retrieved context
    messages.append({
        "role": "user",
        "content": f"Retrieved Context:\n{formatted_context}\nUser Query: {user_query}"
    })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,  # Low temperature to ensure high grounding and minimal creativity
            max_tokens=2400,
            timeout=60.0  # Increased to 60 seconds to allow slow endpoints to respond
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Generation Failed: {e}"

# --- STEP 6: DEDUPLICATED & GROUPED CITATION AND REFERENCE FORMATTING ---
def format_response_with_citations(llm_response, context_chunks):
    """
    Parses the LLM response to check which citation numbers ([1], [2], etc.) are used,
    and appends a clean bibliography. Duplicate sources are grouped together (e.g. [1, 2] Source: ...).
    """
    # Find all digits inside square brackets in the LLM response
    used_indices = set(map(int, re.findall(r'\[(\d+)\]', llm_response)))
    
    # If no citations are used, return response as is
    if not used_indices:
        return llm_response
        
    # Group indices and sections by unique source key (source_name, title, url)
    grouped_refs = {}
    for idx in sorted(used_indices):
        chunk_idx = idx - 1
        if 0 <= chunk_idx < len(context_chunks):
            chunk = context_chunks[chunk_idx]
            meta = chunk.get("metadata", {})
            source_name = meta.get("source_name", "Unknown Source")
            title = meta.get("title", "Unknown Title")
            url = meta.get("source_url", "")
            
            # Extract section info
            section = meta.get("section_name", meta.get("section", ""))
            
            ref_key = (source_name, title, url)
            if ref_key not in grouped_refs:
                grouped_refs[ref_key] = {"indices": [], "sections": set()}
                
            grouped_refs[ref_key]["indices"].append(idx)
            if section:
                grouped_refs[ref_key]["sections"].add(section)
                
    references = []
    # Sort the bibliography by the first citation number of each group (e.g. [1, 3] appears before [2])
    sorted_groups = sorted(grouped_refs.items(), key=lambda x: x[1]["indices"][0])
    
    for ref_key, data in sorted_groups:
        source_name, title, url = ref_key
        indices_str = ", ".join(map(str, data["indices"]))
        sections_list = sorted(list(data["sections"]))
        
        ref_str = f"[{indices_str}] Source: {source_name} ({title})"
        if sections_list:
            sections_str = ", ".join(sections_list)
            ref_str += f" - Section(s): {sections_str}"
        if url:
            ref_str += f" - URL: {url}"
        references.append(ref_str)
            
    if not references:
        return llm_response
        
    bibliography = "\n\nReferences:\n" + "\n".join(references)
    
    # Place references before the medical disclaimer if present to look clean
    disclaimer_marker = "Disclaimer:"
    if disclaimer_marker in llm_response:
        parts = llm_response.split(disclaimer_marker)
        main_body = parts[0].strip()
        disclaimer = disclaimer_marker + parts[1]
        return f"{main_body}\n{bibliography}\n\n{disclaimer}"
    else:
        return f"{llm_response}\n{bibliography}"

# --- STEP 5: CONNECT RETRIEVAL TO GENERATION ---
def query_rag_chatbot(user_query, chat_history=None, n_results=5, temperature=0.1, language="English"):
    """
    End-to-end RAG Chatbot entrypoint with Multi-Language Support.
    """
    if chat_history is None:
        chat_history = []

    # 1. Run local safety and proper noun checks on the user's raw input query
    from retrieve import is_unsafe_medical_query, passes_proper_noun_check

    # A. Medical Prescription/Dosage Check
    if is_unsafe_medical_query(user_query):
        print("🛑 Local Guardrail Triggered: Medical Prescription/Dosage Check")
        return {
            "answer": "As an AI health awareness assistant, I cannot provide medical diagnoses, drug prescriptions, medicine dosages, or treatment decisions. Please consult a qualified medical professional for specific clinical advice and treatment.",
            "chunks": [],
            "distance": 1.0
        }

    # B. Proper Noun check (verifies user is asking about supported schemes if query is English)
    # Skip proper noun check for non-English queries as they will be translated to English during rewriting
    if (not language or language == "English") and not passes_proper_noun_check(user_query):
        print("🛑 Local Guardrail Triggered: Out-of-Bounds Query (Proper Noun Check)")
        return {
            "answer": "I am sorry, but I do not have enough information in my database to answer your query.",
            "chunks": [],
            "distance": 1.0
        }

    print(f"\n[USER QUERY ({language})] '{user_query}'")

    # 2. Rewrite/Translate the query into standalone English search terms
    search_query = rewrite_query_with_history(user_query, chat_history, language=language)
    
    # 3. Retrieve matching chunks on the English query (bypassing proper noun check for internal rewrite)
    retrieval_result = retrieve_for_generation(search_query, n_results=n_results, run_proper_noun_check=False)
    
    # 4. Handle remaining guardrail rejections (semantic distance check)
    if retrieval_result["status"] == "REJECTED_OUT_OF_BOUNDS":
        print("🛑 Local Guardrail Triggered: Out-of-Bounds Query (Distance)")
        return {
            "answer": retrieval_result["message"],
            "chunks": [],
            "distance": retrieval_result["distance"]
        }
        
    # 5. If retrieval succeeded, feed chunks into Llama 3.1 70B for synthesis in target language
    print(f"✅ Retrieval Succeeded (Distance: {retrieval_result['distance']:.3f}). Calling Llama 3.1 70B ({language})...")
    raw_response = generate_response(user_query, retrieval_result["chunks"], chat_history=chat_history, temperature=temperature, language=language)
    
    # 6. Format citations and references
    formatted_response = format_response_with_citations(raw_response, retrieval_result["chunks"])
    return {
        "answer": formatted_response,
        "chunks": retrieval_result["chunks"],
        "distance": retrieval_result["distance"]
    }

def test_live_rag_pipeline():
    """
    Runs end-to-end RAG checks against the active database, testing both single-turn and multi-turn conversations.
    """
    import time
    
    # 1. Single-Turn Test Queries
    print("[START] Running Single-Turn RAG Chatbot Tests...\n" + "="*85)
    single_queries = [
        "What are the common symptoms of diabetes?",
        "What are the eligibility criteria for women under the PMMVY scheme?"
    ]
    for idx, q in enumerate(single_queries):
        if idx > 0:
            print("\n[INFO] Pausing for 2 seconds to respect API rate limits...")
            time.sleep(2)
        response = query_rag_chatbot(q)
        print("-" * 60)
        print(response["answer"])
        print("="*85)

    # 2. Multi-Turn Test Queries (Conversation Memory)
    print("\n\n[START] Running Multi-Turn Chat History Tests...\n" + "="*85)
    chat_history = []
    conversation = [
        "What is Ayushman Bharat PM-JAY?",
        "Who is eligible to apply for this scheme?",  # LLM must resolve "this scheme" to PM-JAY using history
        "Does it cover outpatient consultations?"     # LLM must resolve "it" to PM-JAY
    ]
    
    for idx, q in enumerate(conversation):
        if idx > 0:
            print("\n[INFO] Pausing for 2 seconds to respect API rate limits...")
            time.sleep(2)
            
        # Call RAG with current chat history
        response = query_rag_chatbot(q, chat_history=chat_history)
        print("-" * 60)
        print(response["answer"])
        print("="*85)
        
        # Append current turn to chat history
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    test_live_rag_pipeline()
