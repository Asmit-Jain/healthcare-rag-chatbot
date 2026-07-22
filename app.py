import os
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
    update_session_title
)

# --- STEP 2: ACTIVE CONNECTION CHECKS ---
def get_chromadb_status():
    try:
        return chroma_client.heartbeat() is not None
    except Exception:
        return False

def get_nvidia_status():
    api_key = os.getenv("NVIDIA_API_KEY")
    return api_key is not None and len(api_key.strip()) > 0

db_active = get_chromadb_status()
nvidia_active = get_nvidia_status()
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

    /* Step 5: Citation chip card hover effects */
    .citation-chip {
        background-color: #1e293b;
        border: 1px solid #10b981;
        border-radius: 6px;
        padding: 4px 10px;
        text-align: center;
        color: #10b981;
        font-weight: 500;
        font-size: 0.8rem;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    .citation-chip:hover {
        background-color: #10b981;
        color: #0b0f19;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- STEP 1: MAIN AREA HEADER ---
st.markdown('<div class="dashboard-title">🏥 MedLink Healthcare RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">A secure, grounded, and verified retrieval-augmented clinical awareness platform.</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# --- STEP 3: MEDICAL DISCLAIMER NOTICE ---
disclaimer_html = """
<div class="disclaimer-card">
    <span style="font-size: 1.2rem; line-height: 1;">⚠️</span>
    <div class="disclaimer-text">
        <strong>Medical Boundaries Disclaimer:</strong> As an AI healthcare awareness dashboard, this platform is designed purely to help navigate official public policy and general health resources. It <strong>cannot</strong> diagnose conditions, prescribe medications, or recommend treatment dosages. For clinical advice, always consult a qualified physician.
    </div>
</div>
"""
st.markdown(disclaimer_html, unsafe_allow_html=True)

# --- STEP 4: SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggestion_clicked" not in st.session_state:
    st.session_state.suggestion_clicked = None
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

# --- STEP 2: SIDEBAR IMPLEMENTATION ---
with st.sidebar:
    # --- CHATGPT-STYLE SESSIONS SIDEBAR ---
    st.markdown("### 💬 Chat History")
    
    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.active_session_id = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.suggestion_clicked = None
        st.rerun()

    # 2. Render past chat sessions from MongoDB Atlas
    if mongo_active:
        past_sessions = get_all_sessions()
        if past_sessions:
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            for sess in past_sessions:
                s_id = sess["session_id"]
                s_title = sess.get("title", "Untitled Chat")
                
                # Truncate title if too long for sidebar
                display_title = s_title[:26] + "..." if len(s_title) > 26 else s_title
                
                # Highlight active session
                is_active = (s_id == st.session_state.active_session_id)
                button_label = f"📌 {display_title}" if is_active else f"💬 {display_title}"
                
                if st.button(button_label, key=f"sess_btn_{s_id}", use_container_width=True):
                    st.session_state.active_session_id = s_id
                    # Load messages from MongoDB for this session
                    loaded_msgs = get_session_messages(s_id)
                    st.session_state.messages = loaded_msgs
                    
                    # Reconstruct chat_history for LLM contextualizer
                    reconstructed_history = []
                    for m in loaded_msgs:
                        clean_text = m["content"].split("\n\nReferences:")[0].strip()
                        reconstructed_history.append({"role": m["role"], "content": clean_text})
                    st.session_state.chat_history = reconstructed_history
                    st.session_state.suggestion_clicked = None
                    st.rerun()
        else:
            st.caption("No saved chats found.")
    else:
        st.caption("⚠️ MongoDB Offline - History Disabled")
        
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Developer Control Panel")
    st.markdown("Configure hyperparameters and inspect live connection diagnostics.")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # 1. Live Connection Status badges
    st.markdown("**🔌 Live Diagnostics**")
    
    db_badge = '<span class="status-badge-green"></span>' if db_active else '<span class="status-badge-red"></span>'
    db_text = '<span style="color:#10b981;">Connected</span>' if db_active else '<span style="color:#ef4444;">Offline</span>'
    
    embedder_badge = '<span class="status-badge-green"></span>' if db_active else '<span class="status-badge-red"></span>'
    embedder_text = '<span style="color:#10b981;">Active</span>' if db_active else '<span style="color:#ef4444;">Inactive</span>'
    
    api_badge = '<span class="status-badge-green"></span>' if nvidia_active else '<span class="status-badge-red"></span>'
    api_text = '<span style="color:#10b981;">Active</span>' if nvidia_active else '<span style="color:#ef4444;">API Key Missing</span>'
    
    mongo_badge = '<span class="status-badge-green"></span>' if mongo_active else '<span class="status-badge-red"></span>'
    mongo_text = '<span style="color:#10b981;">Connected</span>' if mongo_active else '<span style="color:#ef4444;">Offline</span>'
    
    st.markdown(
        f'<div class="diagnostics-text">'
        f'{db_badge} ChromaDB: {db_text}<br>'
        f'{embedder_badge} BGE-M3 Embedder: {embedder_text}<br>'
        f'{api_badge} Llama 3.1 NIM API: {api_text}<br>'
        f'{mongo_badge} MongoDB Atlas: {mongo_text}'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # 2. Dynamic sliders for RAG parameters
    st.markdown("**🛠️ Parameters**")
    n_results = st.slider(
        "Retrieve Count (n_results)",
        min_value=1,
        max_value=10,
        value=5,
        help="Capping database context size dynamically."
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Low values ensure grounding and eliminate hallucinations."
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # 3. Clean Clear Conversation Button
    if st.button("🗑️ Clear Active Conversation", use_container_width=True):
        if st.session_state.active_session_id and mongo_active:
            delete_session(st.session_state.active_session_id)
        st.session_state.active_session_id = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.suggestion_clicked = None
        st.success("Conversation cleared successfully!")
        st.rerun()

# --- STEP 4 & 5: DISPLAY CONVERSATION BUBBLES ---
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Clean up plain-text references for the UI bubble if it's assistant's response
        display_content = message["content"]
        if message["role"] == "assistant" and "\n\nReferences:" in display_content:
            parts = display_content.split("\n\nReferences:")
            answer_part = parts[0]
            disclaimer_part = ""
            # Re-extract disclaimer to display it at the bottom of the clean response
            if "Disclaimer:" in parts[1]:
                disclaimer_text = parts[1].split("Disclaimer:")[1].strip()
                disclaimer_part = f"\n\n*Disclaimer: {disclaimer_text}*"
            display_content = answer_part + disclaimer_part
            
        st.markdown(display_content)
        
        # Step 5: Render Citations and RAG Inspector under assistant response
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
            
            # 2. Render RAG Inspector diagnostics
            if "chunks" in message and message["chunks"]:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔍 RAG Inspector - Retrieval Diagnostics"):
                    st.markdown(f"**Best Semantic Distance:** `{message.get('distance', 1.0):.3f}` (Refusal Threshold: `0.39`)")
                    st.markdown("---")
                    for idx, chunk in enumerate(message["chunks"]):
                        meta = chunk["metadata"]
                        st.markdown(f"**Matched Chunk {idx+1}:** `{chunk['id']}`")
                        st.markdown(f"- **Document:** {meta.get('title')}")
                        st.markdown(f"- **Section:** {meta.get('section', 'N/A')}")
                        st.markdown(f"- **Source Directory:** {meta.get('source_name', 'N/A')}")
                        st.text_area(f"Raw Text Content (Chunk {idx+1})", value=chunk["text"], height=120, disabled=True, key=f"raw_text_{msg_idx}_{chunk['id']}_{idx}")

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
user_query = st.chat_input("Ask MedLink...")

# If suggestion was clicked, intercept and overwrite the input
if st.session_state.suggestion_clicked:
    user_query = st.session_state.suggestion_clicked
    st.session_state.suggestion_clicked = None  # Reset state

if user_query:
    # 0. Auto-create new session in MongoDB Atlas on first query if active_session_id is None
    if st.session_state.active_session_id is None and mongo_active:
        new_sess_id = f"session-{uuid.uuid4()}"
        auto_title = user_query[:28] + "..." if len(user_query) > 28 else user_query
        create_chat_session(new_sess_id, title=auto_title)
        st.session_state.active_session_id = new_sess_id

    # 1. Render and append user's query
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Save user message to MongoDB Atlas
    if st.session_state.active_session_id and mongo_active:
        append_message_to_session(st.session_state.active_session_id, "user", user_query)
    
    # 2. Call backend RAG pipeline inside a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching verified database and calling Llama 3.1..."):
            result = query_rag_chatbot(
                user_query=user_query,
                chat_history=st.session_state.chat_history,
                n_results=n_results,
                temperature=temperature
            )
            answer = result["answer"]
            st.markdown(answer)
            
    # 3. Append assistant response and metadata to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": result["chunks"],
        "distance": result["distance"]
    })
    
    # Save assistant response to MongoDB Atlas (including chunks and distance!)
    if st.session_state.active_session_id and mongo_active:
        append_message_to_session(
            st.session_state.active_session_id,
            "assistant",
            answer,
            chunks=result["chunks"],
            distance=result["distance"]
        )
    
    # 4. Update the sliding multi-turn context memory
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # 5. Refresh page to update view
    st.rerun()
