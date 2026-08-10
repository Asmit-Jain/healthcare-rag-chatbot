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

# Initialize OpenAI client using Groq's LPU API endpoint
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Error: GROQ_API_KEY environment variable is not set in the .env file.")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# Groq LPU Models: Meta Llama 3.3 70B (Primary Synthesis) & Meta Llama 3.1 8B (Fast Translation)
MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_NAME_FAST = "llama-3.1-8b-instant"

# --- STEP 4: SYSTEM PROMPT WITH STRICT GROUNDING AND MEDICAL GUARDRAILS ---
SYSTEM_PROMPT = """You are a professional, cautious, and helpful Healthcare Awareness AI Assistant. Your goal is to answer the user's query by strictly using ONLY the facts provided in the "Retrieved Context" section below.

Follow these strict rules at all times:
1. Grounding: Answer the query using ONLY the provided Retrieved Context. If the answer cannot be found or reasonably inferred from the context, state clearly and politely: "I am sorry, but I do not have enough information to answer your query." Do not use any external knowledge or make up facts.
2. Medical Boundaries: You are NOT a doctor or medical professional. 
   - DO NOT diagnose conditions, prescribe medications, or recommend specific drug dosages or treatment decisions under any circumstances.
   - If the user asks for prescriptions, dosages, or diagnostic medical treatment, you must refuse to answer and advise them to consult a qualified physician.
3. Citation Rule: When you state a fact from a retrieved chunk, you MUST cite it using inline brackets matching its chunk number (e.g., [1], [2]).
4. Tone: Maintain a highly cautious, educational, and objective tone at all times.
5. Direct & Detailed Structure: DO NOT echo, restate, or repeat the user's question at the beginning of your response. Start directly with the answer content. When answering in any language, provide complete, detailed, and thorough structured bullet points rather than brief summaries.
"""

def get_localized_guardrail_refusal(refusal_type, language="English"):
    """
    Returns localized guardrail refusal messages in English, Hindi, Hinglish, Marathi, German, Spanish, French, Bengali, Tamil, Telugu, or Gujarati.
    Falls back to fast LLM translation for custom languages like Japanese, Russian, etc.
    """
    refusals = {
        "medical_safety": {
            "English": "As an AI health awareness assistant, I cannot provide medical diagnoses, drug prescriptions, medicine dosages, or treatment decisions. Please consult a qualified medical professional for specific clinical advice and treatment.",
            "Hindi (हिंदी)": "एक स्वास्थ्य जागरूकता AI सहायक के रूप में, मैं चिकित्सा निदान, दवा के पर्चे, दवा की खुराक या उपचार के निर्णय प्रदान नहीं कर सकता। कृपया विशिष्ट नैदानिक सलाह और उपचार के लिए एक योग्य चिकित्सक से परामर्श करें।",
            "Hinglish (Hindi in Roman script)": "Mai ek health awareness AI assistant hu, isliye mai medical diagnoses, drug prescriptions, medicine dosages ya treatment decisions nahi de sakta. Kripya specific clinical advice aur treatment ke liye ek qualified doctor se consult kare.",
            "Marathi (मराठी)": "आरोग्य जागरूकता AI सहाय्यक म्हणून, मी वैद्यकीय निदान, औषधांची प्रिस्क्रिप्शन, औषधांचे डोस किंवा उपचारांचे निर्णय देऊ शकत नाही. कृपया विशिष्ट क्लिनिकल सल्ला आणि उपचारांसाठी पात्र वैद्यकीय व्यावसायिकांचा सल्ला घ्या.",
            "German (Deutsch)": "Als KI-Assistent für Gesundheitsbewusstsein kann ich keine medizinischen Diagnosen, Verschreibungen, Dosierungen oder Behandlungsentscheidungen anbieten. Bitte konsultieren Sie einen qualifizierten Arzt.",
            "Spanish (Español)": "Como asistente de concienciación sobre la salud por IA, no puedo proporcionar diagnósticos médicos, recetas de medicamentos, dosis ni decisiones de tratamiento. Consulte a un profesional médico cualificado.",
            "French (Français)": "En tant qu'assistant de sensibilisation à la santé par IA, je ne peux pas fournir de diagnostics médicaux, de prescriptions, de dosages ou de décisions de traitement. Veuillez consulter un médecin qualifié.",
            "Bengali (বাংলা)": "একটি স্বাস্থ্য সচেতণতা AI সহকারী হিসেবে, আমি চিকিৎসা সংক্রান্ত রোগ নির্ণয়, ওষুধের প্রেসক্রিপশন, ওষুধের মাত্রা বা চিকিৎসার সিদ্ধান্ত দিতে পারি না। অনুগ্রহ করে একজন যোগ্যতাসম্পন্ন চিকিৎসকের সাথে পরামর্শ করুন।",
            "Tamil (தமிழ்)": "சுகாதார விழிப்புணர்வு AI உதவியாளராக, என்னால் மருத்துவ பரிசோதனை, மருந்துக் குறிப்பு அல்லது சிகிச்சை முடிவுகளை வழங்க முடியாது. தகுதியான மருத்துவரை அணுகவும்.",
            "Telugu (తెలుగు)": "ఆరోగ్య అవగాహన AI సహాయకుడిగా, నేను వైద్య నిర్ధారణలు, ఔషధ ప్రిస్క్రిప్షన్లు లేదా చికిత్స నిర్ణయాలను అందించలేను. దయచేసి అర్హత కలిగిన వైద్యుడిని సంప్రదించండి.",
            "Gujarati (ગુજરાતી)": "આરોગ્ય જાગરૂકતા AI સહાયક તરીકે, હું તબીબી નિદાન, દવાની પ્રિસ્ક્રિપ્શનો કે સારવારના નિર્ણયો આપી શકતો નથી. કૃપા કરીને લાયક તબીબી વ્યાવસાયિકની સલાહ લો."
        },
        "out_of_bounds": {
            "English": "I am sorry, but I do not have enough information in my verified database to answer your query.",
            "Hindi (हिंदी)": "मुझे खेद है, लेकिन मेरे सत्यापित डेटाबेस में आपके प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।",
            "Hinglish (Hindi in Roman script)": "Mujhe khed hai, lekin mere verified database me aapke question ka answer dene ke liye sufficient information nahi hai.",
            "Marathi (मराठी)": "मला वाईट वाटते, पण माझ्या पडताळलेल्या डेटाबेसमध्ये तुमच्या प्रश्नाचे उत्तर देण्यासाठी पुरेशी माहिती नाही.",
            "German (Deutsch)": "Es tut mir leid, aber in meiner verifizierten Datenbank sind nicht genügend Informationen vorhanden, um Ihre Anfrage zu beantworten.",
            "Spanish (Español)": "Lo siento, pero no tengo suficiente información en mi base de datos verificada para responder a su consulta.",
            "French (Français)": "Je suis désolé, mais je ne dispose pas de suffisamment d'informations dans ma base de données vérifiée pour répondre à votre demande.",
            "Bengali (বাংলা)": "আমি দুঃখিত, কিন্তু আমার নিবন্ধিত ডেটাবেসে আপনার প্রশ্নের উত্তর দেওয়ার জন্য পর্যাপ্ত তথ্য নেই।",
            "Tamil (தமிழ்)": "மன்னிக்கவும், உங்கள் கேள்விக்கு பதிலளிக்க எனது சரிபார்க்கப்பட்ட தரவுத்தளத்தில் போதுமான தகவல் இல்லை.",
            "Telugu (తెలుగు)": "క్షమించండి, మీ ప్రశ్నకు సమాధానం ఇవ్వడానికి నా నిరూపిత డేటాబేస్‌లో తగినంత సమాచారం లేదు.",
            "Gujarati (ગુજરાતી)": "મને ખેદ છે, પરંતુ મારા ચકાસાયેલ ડેટાબેઝમાં તમારા પ્રશ્નનો જવાબ આપવા માટે પૂરતી માહિતી નથી."
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

def extract_clean_search_keywords(user_query: str) -> str:
    """
    Fallback helper: Strips common non-English / Hinglish stop words to extract core English search keywords
    (e.g., 'Cancer kya hai aur iske symptoms kya hain?' -> 'Cancer symptoms') for accurate ChromaDB vector retrieval.
    """
    stopwords = {
        'kya', 'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'me', 'mein', 'par', 'aur', 'ya',
        'iske', 'iski', 'iska', 'inhe', 'unhe', 'kaise', 'kab', 'kyun', 'kahan', 'bhi', 'hoga',
        'hogi', 'hote', 'hoti', 'hota', 'chahiye', 'batao', 'bataiye', 'tell', 'about', 'what',
        'is', 'are', 'the', 'que', 'es', 'est', 'le', 'la', 'les', 'de', 'du', 'des', 'und', 'ist'
    }
    words = re.findall(r'\b[a-zA-Z0-9-]+\b', user_query)
    clean_tokens = [w for w in words if w.lower() not in stopwords]
    
    if clean_tokens:
        clean_query = " ".join(clean_tokens)
        print(f"🧹 Smart Keyword Extraction Fallback: Extracted search terms: '{clean_query}'")
        return clean_query
    return user_query

def rewrite_query_with_history(user_query, chat_history, language="English"):
    """
    Fast query contextualization & translation engine:
    1. Pipeline Bypass: Bypasses LLM call entirely for initial English queries (saves 20-45s).
    2. Fast 8B Handoff: Uses Llama 3.1 8B (MODEL_NAME_FAST) for rapid translation/contextualization (0.8-1.5s).
    """
    clean_lang = normalize_language_name(language)
    
    # Strategy 1: Pipeline Bypass Optimization for initial English queries
    if not chat_history and (not clean_lang or clean_lang == "English"):
        print(f"⚡ Pipeline Bypass (Strategy 1): Initial English query detected. Skipping LLM translation call.")
        return user_query

    # Assemble sanitized conversation history snippet for prompt
    history_str = ""
    if chat_history:
        for msg in chat_history[-10:]:  # Limit to last 10 messages (5 turns)
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            
            # Sanitize content: strip references, disclaimers across all languages, and bracket citations [1], [2]
            clean_content = content.split("References:\n")[0].split("\n\nReferences:")[0]
            disclaimer_pattern = r'(\*?\b(Disclaimer|Haftungsausschluss|Descargo de responsabilidad|Avertissement|अस्वीकरण|डिस्क्लोमर|डिस्क्लेमर|দাবি পরিত্যাগ|மறுப்பு|గమనిక|અસ્વીકરણ)\b:?.*$)'
            clean_content = re.sub(disclaimer_pattern, '', clean_content, flags=re.IGNORECASE | re.DOTALL)
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

    # Strategy 2: Fast 8B Model (MODEL_NAME_FAST) for rapid 0.8-1.5s translation/contextualization
    try:
        response_fast = client.chat.completions.create(
            model=MODEL_NAME_FAST,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            timeout=6.0
        )
        raw_rewritten = response_fast.choices[0].message.content.strip()
        rewritten = parse_rewritten_query(raw_rewritten)
        print(f"⚡ Fast Multilingual Context (Strategy 2 - 8B Fast Model): Rewrote query as: '{rewritten}'")
        return rewritten
    except Exception as e_8b:
        print(f"[WARNING] 8B Fast Model query translation timed out or failed: {e_8b}. Falling back to Keyword Extraction.")
        return extract_clean_search_keywords(user_query)

def parse_rewritten_query(raw_text: str) -> str:
    """
    Utility parser to extract clean standalone query text from LLM completion responses.
    """
    if "Standalone English Search Query:" in raw_text:
        return raw_text.split("Standalone English Search Query:")[-1].strip().strip('"')
    elif "Translation:" in raw_text:
        return raw_text.split("Translation:")[-1].strip().strip('"')
    else:
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        return lines[-1].strip('"') if lines else raw_text

def normalize_language_name(language: str) -> str:
    """
    Strips non-Latin script parenthetical suffixes from UI selectbox strings to provide
    clean language instructions to Llama 3.1 70B (e.g. 'Gujarati (ગુજરાતી)' -> 'Gujarati').
    Works universally across all world languages.
    """
    if not language:
        return "English"
    if "Hinglish" in language:
        return "Hinglish (Hindi written using English/Roman alphabet)"
    clean_name = language.split("(")[0].strip()
    return clean_name if clean_name else language

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

    # Normalize language string to prevent model confusion
    clean_lang = normalize_language_name(language)

    # Dynamic System Prompt with Language Constraint
    sys_prompt = SYSTEM_PROMPT
    if clean_lang and clean_lang != "English":
        sys_prompt += f"\n6. STRICT LANGUAGE REQUIREMENT: Regardless of the language of the user's question or previous chat messages, you MUST write your ENTIRE final response using complete, fluent sentences strictly in {clean_lang}. Retain inline citation brackets like [1], [2] intact."

    # Compile messages payload starting with system instructions
    messages = [{"role": "system", "content": sys_prompt}]

    # Append sanitized chat history (limit to last 10 messages / 5 turns)
    for msg in chat_history[-10:]:
        content = msg["content"]
        if msg["role"] == "assistant":
            # Strip references, disclaimers across all languages, and bracket numbers to save prompt tokens
            content = content.split("References:\n")[0].split("\n\nReferences:")[0]
            disclaimer_pattern = r'(\*?\b(Disclaimer|Haftungsausschluss|Descargo de responsabilidad|Avertissement|अस्वीकरण|डिस्क्लोमर|डिस्क्लेमर|দাবি পরিত্যাগ|மறுப்பு|గమనిక|અસ્વીકરણ)\b:?.*$)'
            content = re.sub(disclaimer_pattern, '', content, flags=re.IGNORECASE | re.DOTALL).strip()
        messages.append({
            "role": msg["role"],
            "content": content
        })

    user_payload = f"Retrieved Context:\n{formatted_context}\nUser Query: {user_query}"
    if clean_lang and clean_lang != "English":
        user_payload += f"\n\nSTRICT LANGUAGE DIRECTIVE: Regardless of previous chat messages or the language of the user's input question, you MUST write your ENTIRE response using complete, fluent sentences strictly in {clean_lang}. Do NOT use any other language or script."

    # Append current turn user query along with retrieved context
    messages.append({
        "role": "user",
        "content": user_payload
    })

    try:
        # Prevent token repetition loops in non-Latin scripts (Korean, Chinese, Japanese) by enforcing min temperature 0.3 & frequency penalty
        gen_temp = max(0.3, temperature) if clean_lang and clean_lang != "English" else temperature

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=gen_temp,
            frequency_penalty=0.3,
            max_tokens=2400,
            timeout=60.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] Generation Failed: {e}"

def generate_response_stream(user_query, context_chunks, chat_history=None, temperature=0.1, language="English"):
    """
    Generator function that streams tokens live from Groq API for Streamlit st.write_stream().
    """
    if chat_history is None:
        chat_history = []

    formatted_context = ""
    for idx, chunk in enumerate(context_chunks):
        meta = chunk.get("metadata", {})
        source_name = meta.get("source_name", "Unknown Source")
        doc_title = meta.get("title", "Unknown Title")
        formatted_context += f"---\n[{idx + 1}] Source: {source_name} ({doc_title})\n"
        formatted_context += f"Content: {chunk.get('text', '')}\n"
    formatted_context += "---\n"

    clean_lang = normalize_language_name(language)
    sys_prompt = SYSTEM_PROMPT
    if clean_lang and clean_lang != "English":
        sys_prompt += f"\n6. STRICT LANGUAGE REQUIREMENT: Regardless of the language of the user's question or previous chat messages, you MUST write your ENTIRE final response using complete, fluent sentences strictly in {clean_lang}. Retain inline citation brackets like [1], [2] intact."

    messages = [{"role": "system", "content": sys_prompt}]
    for msg in chat_history[-10:]:
        content = msg["content"]
        if msg["role"] == "assistant":
            content = content.split("References:\n")[0].split("\n\nReferences:")[0]
            disclaimer_pattern = r'(\*?\b(Disclaimer|Haftungsausschluss|Descargo de responsabilidad|Avertissement|अस्वीकरण|डिस्क्लोमर|डिस्क्लेमर|দাবি পরিত্যাগ|மறுப்பு|గమనిక|અસ્વીકરણ)\b:?.*$)'
            content = re.sub(disclaimer_pattern, '', content, flags=re.IGNORECASE | re.DOTALL).strip()
        messages.append({"role": msg["role"], "content": content})

    user_payload = f"Retrieved Context:\n{formatted_context}\nUser Query: {user_query}"
    if clean_lang and clean_lang != "English":
        user_payload += f"\n\nSTRICT LANGUAGE DIRECTIVE: Regardless of previous chat messages or the language of the user's input question, you MUST write your ENTIRE response using complete, fluent sentences strictly in {clean_lang}. Do NOT use any other language or script."

    messages.append({"role": "user", "content": user_payload})
    gen_temp = max(0.3, temperature) if clean_lang and clean_lang != "English" else temperature

    try:
        response_stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=gen_temp,
            frequency_penalty=0.3,
            max_tokens=2400,
            stream=True,
            timeout=60.0
        )
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n[ERROR] Generation Failed: {e}"

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
def query_rag_chatbot(user_query, chat_history=None, n_results=5, temperature=0.1, language="English", stream=False):
    """
    End-to-end RAG Chatbot entrypoint with Multi-Language Support and optional token streaming.
    """
    if chat_history is None:
        chat_history = []

    # 1. Run local safety and proper noun checks on the user's raw input query
    from retrieve import is_unsafe_medical_query, passes_proper_noun_check

    # A. Medical Prescription/Dosage Check
    if is_unsafe_medical_query(user_query):
        print("🛑 Local Guardrail Triggered: Medical Prescription/Dosage Check")
        refusal_msg = get_localized_guardrail_refusal("medical_safety", language=language)
        if stream:
            def static_stream(): yield refusal_msg
            return {
                "answer": refusal_msg,
                "stream_generator": static_stream(),
                "chunks": [],
                "distance": 1.0
            }
        return {
            "answer": refusal_msg,
            "chunks": [],
            "distance": 1.0
        }

    # B. Proper Noun check (verifies user is asking about supported schemes if query is English)
    if (not language or language == "English") and not passes_proper_noun_check(user_query):
        print("🛑 Local Guardrail Triggered: Out-of-Bounds Query (Proper Noun Check)")
        refusal_msg = get_localized_guardrail_refusal("out_of_bounds", language=language)
        if stream:
            def static_stream(): yield refusal_msg
            return {
                "answer": refusal_msg,
                "stream_generator": static_stream(),
                "chunks": [],
                "distance": 1.0
            }
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
        if stream:
            def static_stream(): yield refusal_msg
            return {
                "answer": refusal_msg,
                "stream_generator": static_stream(),
                "chunks": [],
                "distance": retrieval_result["distance"]
            }
        return {
            "answer": refusal_msg,
            "chunks": [],
            "distance": retrieval_result["distance"]
        }
        
    # 5. If retrieval succeeded, feed chunks into Llama 3.3 70B via Groq API
    print(f"✅ Retrieval Succeeded (Distance: {retrieval_result['distance']:.3f}). Calling Groq LPU API ({language})...")
    
    if stream:
        stream_gen = generate_response_stream(user_query, retrieval_result["chunks"], chat_history=chat_history, temperature=temperature, language=language)
        return {
            "stream_generator": stream_gen,
            "chunks": retrieval_result["chunks"],
            "distance": retrieval_result["distance"]
        }

    raw_response = generate_response(user_query, retrieval_result["chunks"], chat_history=chat_history, temperature=temperature, language=language)
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
