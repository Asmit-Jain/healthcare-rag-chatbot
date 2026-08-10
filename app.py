import os
import re
import uuid
import streamlit as st
from retrieve import chroma_client
from generate import query_rag_chatbot
from db import (
    check_mongo_connection,
    get_all_sessions,
    get_session_messages,
    delete_session,
    create_chat_session,
    append_message_to_session,
    update_session_title,
    toggle_pin_session,
    update_session_language,
    get_session_language
)

# --- STEP 2: ACTIVE CONNECTION CHECKS ---
def get_chromadb_status():
    try:
        return chroma_client.heartbeat() is not None
    except Exception:
        return False

def get_groq_status():
    api_key = os.getenv("GROQ_API_KEY")
    return api_key is not None and len(api_key.strip()) > 0

db_active = get_chromadb_status()
groq_active = get_groq_status()
mongo_active = check_mongo_connection()

# --- STEP 1: PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MedLink RAG Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STEP 1, 2, 3 & 5: PREMIUM CUSTOM CSS THEME & ANIMATIONS ---
custom_css = """
<style>
    /* Import modern Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply clean font family globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    /* Style the main dashboard title */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .dashboard-subtitle {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    /* Sidebar Glassmorphic Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    /* Styled dividers */
    .custom-divider {
        height: 1px;
        background-color: #1f2937;
        margin: 1rem 0;
    }

    /* Step 2: Pulsing green animation */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Step 2: Pulsing red animation */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .status-badge-green {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
        animation: pulse-green 2s infinite;
    }
    
    .status-badge-red {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #ef4444;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
        animation: pulse-red 2s infinite;
    }

    .diagnostics-text {
        font-size: 0.9rem;
        color: #d1d5db;
        line-height: 1.6;
    }

    /* Step 3: Premium Amber Medical Warning Card */
    .disclaimer-card {
        background-color: rgba(245, 158, 11, 0.05);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-start;
    }

    .disclaimer-text {
        color: #f59e0b;
        font-size: 0.92rem;
        line-height: 1.5;
        margin-left: 0.5rem;
    }

    /* Step 3: suggestion cards styles */
    .suggestion-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.75rem;
    }

    /* Step 5: Citation chip card hover effects with Fade-In Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .citation-chip {
        background-color: #1e293b;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 5px 12px;
        text-align: center;
        color: #10b981;
        font-weight: 500;
        font-size: 0.82rem;
        transition: all 0.25s ease-in-out;
        cursor: pointer;
        animation: fadeIn 0.4s ease-out forwards;
    }
    .citation-chip:hover {
        background-color: #10b981;
        color: #0b0f19;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
        transform: scale(1.03);
    }

    /* Glassmorphic Chat Message Cards */
    div[data-testid="stChatMessage"] {
        background-color: rgba(17, 24, 39, 0.65) !important;
        border: 1px solid rgba(31, 41, 55, 0.8) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(12px) !important;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
    }
    div[data-testid="stChatMessage"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(16, 185, 129, 0.4) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.15) !important;
    }

    /* Compact 1-Line Amber Medical Warning Banner */
    .compact-disclaimer {
        background-color: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 1.2rem;
        font-size: 0.85rem;
        color: #f59e0b;
        line-height: 1.4;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Sidebar Button Hover Slide Transition */
    section[data-testid="stSidebar"] div.stButton > button {
        transition: all 0.2s ease-in-out !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        transform: translateX(3px) !important;
        border-color: #10b981 !important;
    }

    /* 3-Dots Popover Styling Adjustments */
    div[data-testid="stPopover"] button svg {
        display: none !important;
    }
    div[data-testid="stPopover"] button {
        padding-left: 2px !important;
        padding-right: 2px !important;
    }
    div[data-testid="stPopoverBody"] {
        padding: 10px 14px !important;
        border-radius: 10px !important;
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
    }
    div[data-testid="stPopoverBody"] button {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

from auth_ui import render_auth_card, render_user_profile_badge

# --- STEP 4: SESSION STATE INITIALIZATION & AUTH ROUTER ---
if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggestion_clicked" not in st.session_state:
    st.session_state.suggestion_clicked = None
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"
if "custom_language" not in st.session_state:
    st.session_state.custom_language = ""

# If user is unauthenticated, render 2-column Auth Portal and stop execution
if st.session_state.authenticated_user is None:
    render_auth_card()
    st.stop()

# --- STEP 1: MAIN AREA HEADER (AUTHENTICATED ONLY) ---
st.markdown('<div class="dashboard-title">🏥 MedLink AI</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Grounded Clinical Awareness & Verified Multilingual Health Intelligence</div>', unsafe_allow_html=True)

# --- STEP 3: COMPACT MEDICAL DISCLAIMER NOTICE ---
disclaimer_html = """
<div class="compact-disclaimer">
    <span style="font-size: 1.1rem; line-height: 1;">⚠️</span>
    <div>
        <strong>Medical Boundaries Disclaimer:</strong> This platform is designed purely for public health policy and awareness navigation. It <strong>cannot</strong> diagnose conditions, prescribe medications, or recommend treatments. Consult a qualified doctor for clinical advice.
    </div>
</div>
"""
st.markdown(disclaimer_html, unsafe_allow_html=True)

current_user_id = st.session_state.authenticated_user.get("email") if st.session_state.authenticated_user else ""

# --- STEP 2: SIDEBAR IMPLEMENTATION ---
with st.sidebar:
    # 0. User Profile Badge
    render_user_profile_badge(st.session_state.authenticated_user)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # --- CHATGPT-STYLE SESSIONS SIDEBAR ---
    st.markdown("### 💬 Chat History")
    
    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.active_session_id = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.suggestion_clicked = None
        st.rerun()

    # 2. Render past chat sessions from MongoDB Atlas with ChatGPT 3-Dots Menu
    if mongo_active:
        past_sessions = get_all_sessions(user_id=current_user_id)
        if past_sessions:
            pinned_sessions = [s for s in past_sessions if s.get("is_pinned", False)]
            recent_sessions = [s for s in past_sessions if not s.get("is_pinned", False)]
            
            def render_session_item(sess):
                s_id = sess["session_id"]
                s_title = sess.get("title", "Untitled Chat")
                is_pinned = sess.get("is_pinned", False)
                is_active = (s_id == st.session_state.active_session_id)
                
                display_title = s_title[:20] + "..." if len(s_title) > 20 else s_title
                btn_icon = "📌" if is_pinned else "💬"
                btn_label = f"{btn_icon} {display_title}"
                
                c_main, c_opts = st.columns([0.80, 0.20])
                with c_main:
                    if st.button(btn_label, key=f"sess_btn_{s_id}", use_container_width=True):
                        st.session_state.active_session_id = s_id
                        loaded_msgs = get_session_messages(s_id)
                        st.session_state.messages = loaded_msgs
                        sess_lang = get_session_language(s_id)
                        st.session_state.selected_language = sess_lang
                        reconstructed_history = []
                        for m in loaded_msgs:
                            clean_text = m["content"].split("\n\nReferences:")[0].strip()
                            reconstructed_history.append({"role": m["role"], "content": clean_text})
                        st.session_state.chat_history = reconstructed_history
                        st.session_state.suggestion_clicked = None
                        st.rerun()
                with c_opts:
                    with st.popover("⋮", use_container_width=True):
                        st.markdown("<div style='font-weight: 600; font-size: 0.9rem; margin-bottom: 6px;'>Chat Options</div>", unsafe_allow_html=True)
                        pin_label = "📌 Unpin Chat" if is_pinned else "📌 Pin Chat"
                        if st.button(pin_label, key=f"pin_{s_id}", use_container_width=True):
                            toggle_pin_session(s_id)
                            st.rerun()
                            
                        new_name = st.text_input("✏️ Rename (Press Enter)", value=s_title, key=f"rename_in_{s_id}")
                        if new_name.strip() and new_name.strip() != s_title:
                            update_session_title(s_id, new_name.strip())
                            st.rerun()
                                
                        if st.button("🗑️ Delete Chat", key=f"del_{s_id}", use_container_width=True):
                            delete_session(s_id)
                            if st.session_state.active_session_id == s_id:
                                st.session_state.active_session_id = None
                                st.session_state.messages = []
                                st.session_state.chat_history = []
                                st.session_state.suggestion_clicked = None
                            st.rerun()

            if pinned_sessions:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #9ca3af; margin-top: 12px; margin-bottom: 4px;'>📌 PINNED CHATS</div>", unsafe_allow_html=True)
                for sess in pinned_sessions:
                    render_session_item(sess)
                    
            if recent_sessions:
                st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #9ca3af; margin-top: 12px; margin-bottom: 4px;'>💬 RECENT CHATS</div>", unsafe_allow_html=True)
                for sess in recent_sessions:
                    render_session_item(sess)
        else:
            st.caption("No saved chats found.")
    else:
        st.caption("⚠️ MongoDB Offline - History Disabled")
        
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.authenticated_user = None
        st.session_state.auth_token = None
        st.session_state.active_session_id = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.suggestion_clicked = None
        st.rerun()

# Optimal Fixed RAG Hyperparameters
n_results = 5
temperature = 0.1

def get_disclaimer_for_language(lang):
    """
    Hybrid Smart Disclaimer System:
    1. Instant lookup for top languages (0.00s latency).
    2. Dynamic LLM translation fallback for custom languages with caching.
    """
    disclaimers_map = {
        "English": "Disclaimer: This information is for educational purposes only. Please consult a qualified medical professional for specific clinical advice and treatment.",
        "Hindi (हिंदी)": "डिस्क्लेमर: यह जानकारी केवल शैक्षिक उद्देश्यों के लिए है। कृपया विशिष्ट नैदानिक सलाह और उपचार के लिए एक योग्य चिकित्सक से परामर्श करें।",
        "Hinglish (Hindi in Roman script)": "Disclaimer: Ye jankari educational purposes ke liye hai. Specific clinical advice aur treatment ke liye ek qualified doctor se consult kare.",
        "Marathi (मराठी)": "अस्वीकरण: ही माहिती केवळ शैक्षणिक हेतूसाठी आहे. कृपया विशिष्ट वैद्यकीय सल्ल्यासाठी आणि उपचारांसाठी पात्र डॉक्टरांचा सल्ला घ्या.",
        "German (Deutsch)": "Haftungsausschluss: Diese Informationen dienen nur zu Bildungszwecken. Bitte konsultieren Sie einen qualifizierten medizinischen Fachmann für spezifische klinische Beratung und Behandlung.",
        "Spanish (Español)": "Descargo de responsabilidad: Esta información es solo para fines educativos. Consulte a un profesional médico cualificado para obtener asesoramiento clínico y tratamiento específicos.",
        "French (Français)": "Avertissement: Ces informations sont fournies à des fins éducatives uniquement. Veuillez consulter un professionnel de la santé qualifié.",
        "Bengali (বাংলা)": "দাবি পরিত্যাগ: এই তথ্য শুধুমাত্র শিক্ষামূলক উদ্দেশ্যে। নির্দিষ্ট চিকিৎসার পরামর্শ ও চিকিৎসার জন্য দয়া করে একজন যোগ্যতাসম্পন্ন চিকিৎসকের সাথে পরামর্শ করুন.",
        "Tamil (தமிழ்)": "மறுப்பு: இந்த தகவல் கல்வி நோக்கங்களுக்காக மட்டுமே. குறிப்பிட்ட மருத்துவ ஆலோசனை மற்றும் சிகிச்சைக்கு தகுதியுள்ள மருத்துவரை அணுகவும்.",
        "Telugu (తెలుగు)": "గమనిక: ఈ సమాచారం కేవలం విద్యా ప్రయోజనాల కోసం మాత్రమే. నిర్దిష్ట వైద్య సలహా మరియు చికిత్స కోసం దయచేసి అర్హత కలిగిన వైద్యుడిని సంప్రదించండి.",
        "Gujarati (ગુજરાતી)": "અસ્વીકરણ: આ માહિતી ફક્ત શૈક્ષણિક હેતુઓ માટે છે. ચોક્કસ તબીબી સલાહ અને સારવાર માટે કૃપા કરીને લાયક તબીબી વ્યાવસાયિકની સલાહ લો."
    }

    if lang in disclaimers_map:
        return disclaimers_map[lang]

    for key, text in disclaimers_map.items():
        if key.split(" ")[0].lower() in lang.lower():
            return text

    if "custom_disclaimer_cache" not in st.session_state:
        st.session_state.custom_disclaimer_cache = {}

    if lang in st.session_state.custom_disclaimer_cache:
        return st.session_state.custom_disclaimer_cache[lang]

    try:
        from generate import client, MODEL_NAME
        english_disclaimer = disclaimers_map["English"]
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"Translate the following medical disclaimer statement into {lang}. Return ONLY the translated disclaimer statement without quotes or extra preamble text:\n\n{english_disclaimer}"
            }],
            temperature=0.0,
            max_tokens=100,
            timeout=10.0
        )
        translated = response.choices[0].message.content.strip().strip('"')
        st.session_state.custom_disclaimer_cache[lang] = translated
        return translated
    except Exception as e:
        print(f"[WARNING] Dynamic disclaimer translation failed: {e}. Falling back to English.")
        return disclaimers_map["English"]

# --- STEP 4 & 5: DISPLAY CONVERSATION BUBBLES ---
for msg_idx, message in enumerate(st.session_state.messages):
    display_content = message.get("content", "")
    if not display_content or not str(display_content).strip():
        continue  # Skip empty messages to prevent blank icon bubbles

    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Strip plain-text references section from display content
            if "\n\nReferences:" in display_content:
                display_content = display_content.split("\n\nReferences:")[0].strip()
            if "References:\n" in display_content:
                display_content = display_content.split("References:\n")[0].strip()
            
            # Universal Disclaimer Stripper: remove any embedded LLM disclaimers across all languages
            disclaimer_pattern = r'(\*?\b(Disclaimer|Haftungsausschluss|Descargo de responsabilidad|Avertissement|अस्वीकरण|डिस्क्लोमर|डिस्क्लेमर|দাবি পরিত্যাগ|மறுப்பு|గమనిక|અસ્વીકરણ)\b:?.*$)'
            display_content = re.sub(disclaimer_pattern, '', display_content, flags=re.IGNORECASE | re.DOTALL).strip()

            # Programmatically append clean localized disclaimer at bottom using message's language
            msg_lang = message.get("language", st.session_state.selected_language)
            disclaimer_text = get_disclaimer_for_language(msg_lang)
            display_content += f"\n\n*{disclaimer_text}*"
            
        st.markdown(display_content)
        
        # Step 5: Render Citations under assistant response
        if message["role"] == "assistant":
            # 1. Parse and render Clickable Citation Links
            if "chunks" in message and message["chunks"]:
                import re
                # Find all citation indices actually used by the LLM in the response text
                used_indices = set(map(int, re.findall(r'\[(\d+)\]', message["content"])))
                
                grouped_sources = {}
                for idx, chunk in enumerate(message["chunks"]):
                    chunk_num = idx + 1
                    # Only map and display the source if the citation index was actually used
                    if chunk_num in used_indices:
                        meta = chunk["metadata"]
                        title = meta.get("title", "Unknown Source")
                        url = meta.get("source_url", "")
                        if url:
                            if url not in grouped_sources:
                                grouped_sources[url] = {"title": title, "indices": []}
                            grouped_sources[url]["indices"].append(chunk_num)
                
                if grouped_sources:
                    st.markdown("<br>**🔗 Clickable Citation Links:**", unsafe_allow_html=True)
                    chips_html = '<div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:5px;">'
                    for url, data in grouped_sources.items():
                        indices_str = ", ".join(map(str, data["indices"]))
                        chips_html += (
                            f'<a href="{url}" target="_blank" style="text-decoration:none;">'
                            f'<div class="citation-chip">'
                            f'[{indices_str}] {data["title"]}'
                            f'</div></a>'
                        )
                    chips_html += '</div>'
                    st.markdown(chips_html, unsafe_allow_html=True)
            


# --- STEP 3: CLICKABLE SUGGESTION CARDS (COLD START) ---
# We only display suggestions if the chat hasn't started yet
if len(st.session_state.messages) == 0:
    st.markdown('<div class="suggestion-header">💡 Quick Start - Sample Queries</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(
            "What is Ayushman Bharat PM-JAY?\n\n*Click to search this scheme*",
            use_container_width=True
        ):
            st.session_state.suggestion_clicked = "What is Ayushman Bharat PM-JAY?"
            st.rerun()
            
    with col2:
        if st.button(
            "What are the common symptoms of diabetes?\n\n*Click to check clinical facts*",
            use_container_width=True
        ):
            st.session_state.suggestion_clicked = "What are the common symptoms of diabetes?"
            st.rerun()
            
    with col3:
        if st.button(
            "What are the eligibility criteria under PMMVY?\n\n*Click to check maternal rules*",
            use_container_width=True
        ):
            st.session_state.suggestion_clicked = "What are the eligibility criteria for women under the PMMVY scheme?"
            st.rerun()

# --- STEP 4: CONVERSATIONAL CHAT INPUT & RUN PIPELINE ---
st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)

# Single Clean Language Selector Dropdown in Main Area
col_lang1, col_lang2 = st.columns([2.5, 1.5])
with col_lang2:
    language_options = [
        "English",
        "Hindi (हिंदी)",
        "Hinglish (Hindi in Roman script)",
        "Bengali (বাংলা)",
        "Tamil (தமிழ்)",
        "Telugu (తెలుగు)",
        "Marathi (मराठी)",
        "Gujarati (ગુજરાતી)",
        "Spanish (Español)",
        "French (Français)",
        "German (Deutsch)",
        "Custom / Other"
    ]
    current_lang = st.session_state.get("selected_language", "English")
    if current_lang in language_options:
        default_lang_idx = language_options.index(current_lang)
    else:
        default_lang_idx = language_options.index("Custom / Other")
        if not st.session_state.get("custom_language") and current_lang not in language_options:
            st.session_state.custom_language = current_lang

    def on_language_select_change():
        chosen = st.session_state.sb_language_select
        if chosen != "Custom / Other":
            st.session_state.selected_language = chosen
            if st.session_state.get("active_session_id") and mongo_active:
                update_session_language(st.session_state.active_session_id, chosen)
        else:
            if st.session_state.get("custom_language"):
                st.session_state.selected_language = st.session_state.custom_language

    def on_custom_language_input_change():
        typed = st.session_state.ti_custom_language.strip()
        if typed:
            st.session_state.custom_language = typed
            st.session_state.selected_language = typed
            if st.session_state.get("active_session_id") and mongo_active:
                update_session_language(st.session_state.active_session_id, typed)

    chosen_lang_option = st.selectbox(
        "🌐 Response Language",
        language_options,
        index=default_lang_idx,
        key="sb_language_select",
        on_change=on_language_select_change,
        help="Select target language for AI responses."
    )

    if chosen_lang_option == "Custom / Other":
        custom_val = st.text_input(
            "Specify Custom Language",
            value=st.session_state.get("custom_language", ""),
            placeholder="e.g. Punjabi, Japanese...",
            key="ti_custom_language",
            on_change=on_custom_language_input_change
        )
        active_target_language = st.session_state.get("custom_language", "").strip() or "English"
    else:
        active_target_language = chosen_lang_option

    st.session_state.selected_language = active_target_language

user_query = st.chat_input("Ask MedLink...")

# If suggestion was clicked, intercept and overwrite the input
if st.session_state.suggestion_clicked:
    user_query = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None  # Reset state

if user_query:
    # Capture target language snapshot for active execution turn
    target_lang_turn = st.session_state.selected_language

    # 0. Auto-create new session in MongoDB Atlas on first query if active_session_id is None
    if st.session_state.active_session_id is None and mongo_active:
        new_sess_id = f"session-{uuid.uuid4()}"
        auto_title = user_query[:28] + "..." if len(user_query) > 28 else user_query
        create_chat_session(new_sess_id, user_id=current_user_id, title=auto_title, language=target_lang_turn)
        st.session_state.active_session_id = new_sess_id

    # 1. Render and append user's query
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Save user message to MongoDB Atlas
    if st.session_state.active_session_id and mongo_active:
        append_message_to_session(st.session_state.active_session_id, "user", user_query)
    
    # 2. Call backend RAG pipeline with live streaming and timer
    with st.chat_message("assistant"):
        import time
        start_time = time.time()
        
        result = query_rag_chatbot(
            user_query=user_query,
            chat_history=st.session_state.chat_history,
            n_results=n_results,
            temperature=temperature,
            language=target_lang_turn,
            stream=True
        )
        
        if "stream_generator" in result:
            raw_answer = st.write_stream(result["stream_generator"])
        else:
            raw_answer = result.get("answer", "")
            st.markdown(raw_answer)
            
        exec_time = time.time() - start_time
        
        # Append citations & references
        from generate import format_response_with_citations
        formatted_answer = format_response_with_citations(raw_answer, result["chunks"])
        
        # Append localized disclaimer
        disclaimer_pattern = r'(\*?\b(Disclaimer|Haftungsausschluss|Descargo de responsabilidad|Avertissement|अस्वीकरण|डिस्क्लोमर|डिस्क्लेमर|দাবি পরিত্যাগ|மறுப்பு|గమనిక|અસ્વીકરણ)\b:?.*$)'
        clean_check = re.sub(disclaimer_pattern, '', raw_answer, flags=re.IGNORECASE | re.DOTALL).strip()
        if not clean_check.startswith("⚠️") and not clean_check.startswith("[ERROR]"):
            disclaimer_text = get_disclaimer_for_language(target_lang_turn)
            if disclaimer_text not in formatted_answer:
                formatted_answer += f"\n\n*{disclaimer_text}*"
        
        answer = formatted_answer
            
    # 3. Append assistant response and metadata to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": result["chunks"],
        "distance": result["distance"],
        "execution_time": exec_time,
        "language": target_lang_turn
    })
    
    # Save assistant response to MongoDB Atlas
    if st.session_state.active_session_id and mongo_active:
        append_message_to_session(
            st.session_state.active_session_id,
            "assistant",
            answer,
            chunks=result["chunks"],
            distance=result["distance"],
            language=target_lang_turn
        )
    
    # 4. Update the sliding multi-turn context memory
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # 5. Refresh page to update view
    st.rerun()
