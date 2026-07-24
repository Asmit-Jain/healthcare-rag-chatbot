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

def get_localized_guardrail_refusal(refusal_type, language="English"):
    """
    Returns localized guardrail refusal messages in English, Hindi, Hinglish, or Spanish.
    Falls back to LLM translation for custom languages.
    """
    refusals = {
        "medical_safety": {
            "English": "As an AI health awareness assistant, I cannot provide medical diagnoses, drug prescriptions, medicine dosages, or treatment decisions. Please consult a qualified medical professional for specific clinical advice and treatment.",
            "Hindi (हिंदी)": "एक स्वास्थ्य जागरूकता AI सहायक के रूप में, मैं चिकित्सा निदान, दवा के पर्चे, दवा की खुराक या उपचार के निर्णय प्रदान नहीं कर सकता। कृपया विशिष्ट नैदानिक सलाह और उपचार के लिए एक योग्य चिकित्सक से परामर्श करें।",
            "Hinglish (Hindi in Roman script)": "Mai ek health awareness AI assistant hu, isliye mai medical diagnoses, drug prescriptions, medicine dosages ya treatment decisions nahi de sakta. Kripya specific clinical advice aur treatment ke liye ek qualified doctor se consult kare.",
            "Spanish (Español)": "Como asistente de concienciación sobre la salud por IA, no puedo proporcionar diagnósticos médicos, recetas de medicamentos, dosis ni decisiones de tratamiento. Consulte a un profesional médico cualificado para obtener asesoramiento clínico y tratamiento específicos."
        },
        "out_of_bounds": {
            "English": "I am sorry, but I do not have enough information in my verified database to answer your query.",
            "Hindi (हिंदी)": "मुझे खेद है, लेकिन मेरे सत्यापित डेटाबेस में आपके प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।",
            "Hinglish (Hindi in Roman script)": "Mujhe khed hai, lekin mere verified database me aapke question ka answer dene ke liye sufficient information nahi hai.",
            "Spanish (Español)": "Lo siento, pero no tengo suficiente información en mi base de datos verificada para responder a su consulta."
        }
    }

    type_dict = refusals.get(refusal_type, refusals["out_of_bounds"])
    if language in type_dict:
        return type_dict[language]
    
    # Check simple key matches (e.g. "Hindi" or "Spanish")
    for lang_key, text in type_dict.items():
        if lang_key.lower() in language.lower():
            return text
            
    # For custom / unhandled languages, perform a fast translation call
    english_text = type_dict["English"]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"Translate the following medical refusal statement into {language}. Return ONLY the translated statement without quotes or extra text:\n\n{english_text}"
            }],
            temperature=0.0,
            max_tokens=150,
            timeout=15.0
        )
        return response.choices[0].message.content.strip().strip('"')
    except Exception as e:
        print(f"[WARNING] Localized refusal translation failed: {e}. Falling back to English.")
        return english_text

def rewrite_query_with_history(user_query, chat_history, language="English"):
    """
    Uses Llama 3.1 70B to rewrite a follow-up query into a standalone search query if history exists,
    or translates non-English/Hinglish queries into English for accurate ChromaDB & BM25 retrieval.
    Sanitizes history by removing references, disclaimers, and bracket numbers.
    """
    if not chat_history and (not language or language == "English"):
        return user_query

    # Assemble sanitized conversation history snippet for prompt
    history_str = ""
    if chat_history:
        for msg in chat_history[-10:]:  # Limit to last 10 messages (5 turns)
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            
            # Sanitize content: strip references, disclaimers, and bracket citations [1], [2]
            clean_content = content.split("References:\n")[0].split("\n\nReferences:")[0]
            clean_content = re.sub(r'Disclaimer:.*$', '', clean_content, flags=re.DOTALL)
            clean_content = re.sub(r'अस्वीकरण:.*$', '', clean_content, flags=re.DOTALL)
            clean_content = re.sub(r'\[\d+\]', '', clean_content)
            clean_content = clean_content.strip()
            
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
        raw_rewritten = response.choices[0].message.content.strip()
        if "Standalone English Search Query:" in raw_rewritten:
            rewritten = raw_rewritten.split("Standalone English Search Query:")[-1].strip().strip('"')
        elif "Translation:" in raw_rewritten:
            rewritten = raw_rewritten.split("Translation:")[-1].strip().strip('"')
        else:
            lines = [line.strip() for line in raw_rewritten.split("\n") if line.strip()]
            rewritten = lines[-1].strip('"') if lines else raw_rewritten

        print(f"🔄 Conversational/Multilingual Context: Rewrote query to English standalone as: '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[WARNING] Query rewriting/translation failed: {e}. Falling back to original query.")
        return user_query

def generate_response(user_query, context_chunks, chat_history=None, temperature=0.1, language="English"):
    """
    Assembles context prompt, appends sanitized chat history, calls Llama model, and returns response.
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
        sys_prompt += f"\n5. STRICT LANGUAGE REQUIREMENT: You MUST synthesize and write your entire final response strictly in {language}. Retain inline citation brackets like [1], [2] intact."

    # Compile messages payload starting with system instructions
    messages = [{"role": "system", "content": sys_prompt}]

    # Append sanitized chat history (limit to last 10 messages / 5 turns)
    for msg in chat_history[-10:]:
        content = msg["content"]
        if msg["role"] == "assistant":
            # Strip references, disclaimers, and bracket numbers to save prompt tokens
            content = content.split("References:\n")[0].split("\n\nReferences:")[0]
            content = re.sub(r'Disclaimer:.*$', '', content, flags=re.DOTALL)
            content = re.sub(r'अस्वीकरण:.*$', '', content, flags=re.DOTALL)
            content = content.strip()
        messages.append({
            "role": msg["role"],
            "content": content
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
            temperature=temperature,
            max_tokens=2400,
            timeout=90.0  # Extended to 90 seconds to prevent queue dropouts
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Generation Failed: {e}"

# --- STEP 6: DEDUPLICATED & GROUPED CITATION AND REFERENCE FORMATTING ---
def format_response_with_citations(llm_response, context_chunks):
    """
    Parses the LLM response to check which citation numbers ([1], [2], etc.) are used,
    and appends a clean bibliography.
    """
    used_indices = set(map(int, re.findall(r'\[(\d+)\]', llm_response)))
    
    if not used_indices:
        return llm_response
        
    grouped_refs = {}
    for idx in sorted(used_indices):
        chunk_idx = idx - 1
        if 0 <= chunk_idx < len(context_chunks):
            chunk = context_chunks[chunk_idx]
            meta = chunk.get("metadata", {})
            source_name = meta.get("source_name", "Unknown Source")
            title = meta.get("title", "Unknown Title")
            url = meta.get("source_url", "")
            section = meta.get("section_name", meta.get("section", ""))
            
            ref_key = (source_name, title, url)
            if ref_key not in grouped_refs:
                grouped_refs[ref_key] = {"indices": [], "sections": set()}
                
            grouped_refs[ref_key]["indices"].append(idx)
            if section:
                grouped_refs[ref_key]["sections"].add(section)
                
    references = []
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
        refusal_msg = get_localized_guardrail_refusal("medical_safety", language=language)
        return {
            "answer": refusal_msg,
            "chunks": [],
            "distance": 1.0
        }

    # B. Proper Noun check (verifies user is asking about supported schemes if query is English)
    if (not language or language == "English") and not passes_proper_noun_check(user_query):
        print("🛑 Local Guardrail Triggered: Out-of-Bounds Query (Proper Noun Check)")
        refusal_msg = get_localized_guardrail_refusal("out_of_bounds", language=language)
        return {
            "answer": refusal_msg,
            "chunks": [],
            "distance": 1.0
        }

    print(f"\n[USER QUERY ({language})] '{user_query}'")

    # 2. Rewrite/Translate the query into standalone English search terms
    search_query = rewrite_query_with_history(user_query, chat_history, language=language)
    
    # 3. Retrieve matching chunks on the English query
    retrieval_result = retrieve_for_generation(search_query, n_results=n_results, run_proper_noun_check=False)
    
    # 4. Handle remaining guardrail rejections (semantic distance check)
    if retrieval_result["status"] == "REJECTED_OUT_OF_BOUNDS":
        print("🛑 Local Guardrail Triggered: Out-of-Bounds Query (Distance)")
        refusal_msg = get_localized_guardrail_refusal("out_of_bounds", language=language)
        return {
            "answer": refusal_msg,
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
