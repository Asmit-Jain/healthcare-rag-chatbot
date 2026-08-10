import os
import streamlit as st
from db import register_user, login_user

def get_user_initials(full_name: str, email: str) -> str:
    """
    Computes a 2-letter uppercase avatar initials string from full name or email.
    """
    if full_name and full_name.strip():
        parts = full_name.strip().split()
        if len(parts) >= 2:
            return (parts[0][0] + parts[-1][0]).upper()
        elif len(parts) == 1 and len(parts[0]) >= 2:
            return parts[0][:2].upper()
        elif len(parts) == 1:
            return parts[0][0].upper()
    
    clean_email = email.split("@")[0] if email else "U"
    return clean_email[:2].upper()

def render_user_profile_badge(user_dict: dict):
    """
    Renders a circular avatar initials badge in the sidebar header.
    """
    if not user_dict:
        return
        
    full_name = user_dict.get("full_name", "User")
    email = user_dict.get("email", "")
    initials = get_user_initials(full_name, email)
    
    badge_html = f"""
    <div style="display: flex; align-items: center; gap: 12px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 10px 14px; margin-bottom: 15px;">
        <div style="width: 38px; height: 38px; min-width: 38px; border-radius: 50%; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: #ffffff; font-weight: 700; font-size: 0.95rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 3px 10px rgba(16, 185, 129, 0.35);">
            {initials}
        </div>
        <div style="overflow: hidden; flex: 1;">
            <div style="font-weight: 600; font-size: 0.92rem; color: #f3f4f6; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                {full_name}
            </div>
            <div style="font-size: 0.78rem; color: #9ca3af; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                {email}
            </div>
        </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

def render_auth_card():
    """
    Renders a perfectly aligned 2-column split portal:
    - Left Column: Welcome Title, Subtitle, User-Centric Feature Cards, and Trust Badge.
    - Right Column: Glass Container Card with Login & Sign Up tabs.
    """
    auth_css = """
    <style>
    .auth-left-container {
        padding-top: 0rem;
    }
    .auth-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        margin-top: 0rem;
        line-height: 1.2;
    }
    .auth-subtitle {
        font-size: 0.98rem;
        color: #9ca3af;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }
    .feature-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        transition: all 0.25s ease-in-out;
    }
    .feature-card:hover {
        border-color: #10b981;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
        transform: scale(1.015);
    }
    .feature-icon {
        font-size: 1.25rem;
        line-height: 1;
        margin-top: 2px;
    }
    .feature-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f3f4f6;
        margin-bottom: 2px;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: #9ca3af;
        line-height: 1.4;
    }
    .trust-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 500;
        margin-top: 8px;
    }
    .auth-right-container {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    </style>
    """
    st.markdown(auth_css, unsafe_allow_html=True)

    col_left, col_space, col_right = st.columns([1.15, 0.08, 1.0])

    with col_left:
        left_html = (
            '<div class="auth-left-container">'
            '<div class="auth-title">Welcome to MedLink AI</div>'
            '<div class="auth-subtitle">An intelligent, grounded public health awareness & government healthcare scheme platform.</div>'
            
            '<div class="feature-card">'
            '<div class="feature-icon">🌐</div>'
            '<div><div class="feature-title">Multilingual Support</div><div class="feature-desc">Ask questions in English, Hindi, Hinglish, Marathi, Bengali, Tamil, Telugu, Gujarati, Spanish, French, German, or any custom language with instant AI translation.</div></div>'
            '</div>'
            
            '<div class="feature-card">'
            '<div class="feature-icon">🏛️</div>'
            '<div><div class="feature-title">Official Policy & WHO Guidance</div><div class="feature-desc">100% verified health information sourced directly from official government health schemes (myScheme.gov.in) and WHO disease factsheets.</div></div>'
            '</div>'
            
            '<div class="feature-card">'
            '<div class="feature-icon">🛡️</div>'
            '<div><div class="feature-title">Safe & Responsible Information</div><div class="feature-desc">Built-in clinical boundary guardrails protecting against unsafe drug prescriptions, dosages, or unverified claims.</div></div>'
            '</div>'
            
            '<div class="feature-card">'
            '<div class="feature-icon">⚡</div>'
            '<div><div class="feature-title">Private & Persistent Sessions</div><div class="feature-desc">Secure JWT account isolation with saved conversation history and 1-click pinned chats.</div></div>'
            '</div>'
            
            '<div class="trust-badge"><span class="status-badge-green"></span> Verified Public Health & Government Healthcare Policy Platform</div>'
            '</div>'
        )
        st.markdown(left_html, unsafe_allow_html=True)

    with col_right:
        header_html = (
            '<div class="auth-right-container">'
            '<div style="text-align: center; margin-bottom: 1.2rem; margin-top: 0.2rem;">'
            '<div style="font-size: 1.6rem; font-weight: 700; color: #f3f4f6;">Account Portal</div>'
            '<div style="font-size: 0.88rem; color: #9ca3af; margin-top: 4px;">Sign in to your account or create a new profile to get started.</div>'
            '</div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tab_login:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            login_email = st.text_input("📧 Email Address", key="login_email_input", placeholder="name@example.com")
            login_password = st.text_input("🔑 Password", type="password", key="login_password_input", placeholder="Enter your password")

            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            if st.button("🔓 Sign In", key="btn_submit_login", type="primary", use_container_width=True):
                if not login_email.strip() or not login_password:
                    st.error("Please enter both email address and password.")
                else:
                    success, res = login_user(login_email.strip(), login_password)
                    if success:
                        st.session_state.authenticated_user = res["user"]
                        st.session_state.auth_token = res["token"]
                        st.success(f"Welcome back, {res['user']['full_name']}!")
                        st.rerun()
                    else:
                        st.error(res)

        with tab_signup:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            signup_name = st.text_input("👤 Full Name", key="signup_name_input", placeholder="e.g. Asmit Jain")
            signup_email = st.text_input("📧 Email Address", key="signup_email_input", placeholder="name@example.com")
            signup_password = st.text_input("🔑 Password", type="password", key="signup_password_input", placeholder="At least 6 characters")
            signup_confirm = st.text_input("🔐 Confirm Password", type="password", key="signup_confirm_input", placeholder="Re-enter password")

            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            if st.button("✨ Create Account", key="btn_submit_signup", type="primary", use_container_width=True):
                if not signup_name.strip():
                    st.error("Please enter your full name.")
                elif not signup_email.strip() or "@" not in signup_email:
                    st.error("Please enter a valid email address.")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif signup_password != signup_confirm:
                    st.error("Passwords do not match. Please check and try again.")
                else:
                    success, res = register_user(signup_email.strip(), signup_name.strip(), signup_password)
                    if success:
                        st.session_state.authenticated_user = res["user"]
                        st.session_state.auth_token = res["token"]
                        st.success(f"Account created successfully! Welcome, {res['user']['full_name']}!")
                        st.rerun()
                    else:
                        st.error(res)
        st.markdown('</div>', unsafe_allow_html=True)
