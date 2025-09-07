import streamlit as st
import os
import base64
import google.generativeai as genai
from typing import List, Dict
from html import escape
from dotenv import load_dotenv

# --- INITIALIZATION AND CONFIGURATION ---
load_dotenv()
try:
    GEMINI_KEY = os.getenv("apiKEY")
    if not GEMINI_KEY:
        st.error("API key not found. Please set 'apiKEY' in your .env file.")
        st.stop()
    genai.configure(api_key=GEMINI_KEY)
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --- GEMINI BOT LOGIC ---
SYSTEM_PROMPT_TEMPLATE = (
    "You are MediBot, a soft-spoken, empathetic medical advisory assistant.\n"
    "- Speak in a caring, calm tone.\n"
    "- Your primary role is to provide general, educational information about symptoms and conditions.\n"
    "- DO NOT provide any diagnoses, prescriptions, or definitive medical advice. This is crucial.\n"
    "- If the user asks an unrelated question, reply gently: '⚠️ I can only help with medical-related queries.'\n"
    "- For any medical query, always provide information on self-care measures and strongly recommend consulting a healthcare professional.\n"
    "- Use the requested language: {language}.\n"
    "- End every response with the disclaimer: '⚠️ This is not a substitute for professional medical consultation.'\n\n"
)

def get_medibot_reply(messages: List[Dict], language: str = "English", attachment_b64: str = None) -> str:
    """Calls Gemini and returns the assistant reply as text."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Build the conversation history payload
        conversation_history = []
        conversation_history.append({"role": "user", "parts": [SYSTEM_PROMPT_TEMPLATE.format(language=language)]})
        conversation_history.append({"role": "model", "parts": ["Understood. I will follow all the provided instructions."]})
        
        # Add the conversation history from the session state
        for m in messages:
            if m.get("role") == "user":
                conversation_history.append({"role": "user", "parts": [m.get('content')]})
            elif m.get("role") == "assistant":
                conversation_history.append({"role": "model", "parts": [m.get('content')]})

        # Add attachment to the last user message in the payload if it exists
        if attachment_b64 and conversation_history:
            last_user_message = conversation_history[-1]
            last_user_message["parts"].append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": attachment_b64
                }
            })

        # Use safety settings to prevent content from being filtered too aggressively
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        # Start a chat session with the full payload
        chat = model.start_chat(history=conversation_history)
        
        # Send the latest message from the user
        last_user_message_content = messages[-1].get('content')
        
        response = chat.send_message(last_user_message_content, safety_settings=safety_settings)
        
        if response and response.text:
            return response.text
        else:
            if response and response.prompt_feedback:
                print(f"Content was blocked. Safety ratings: {response.prompt_feedback.safety_ratings}")
                return "I'm sorry, I cannot respond to that query. It may have been flagged for safety reasons."
            return "Sorry, I couldn't generate a response. Please try again."

    except Exception as e:
        st.error(f"Error calling MediBot: {e}") 
        return "An internal error occurred. Please try again."

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="MediBot", layout="wide")

# Custom CSS for chat bubbles to ensure visibility
st.markdown("""
    <style>
    .st-emotion-cache-183-f3p { margin: auto; width: 80%; }
    .stTextInput, .stFileUploader { padding: 10px; }
    .chat-bubble { padding: 12px; border-radius: 10px; margin: 8px 0; white-space: pre-wrap; overflow-wrap: break-word; }
    .user-bubble { background-color: #d1e7dd; color: #155724; }
    .assistant-bubble { background-color: #cce5ff; color: #004085; }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 MediBot: Your Health Assistant")
st.markdown("---")

# --- SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CHAT DISPLAY ---
st.markdown("## Conversation")
chat_col, control_col = st.columns([4, 1])

with chat_col:
    if st.session_state.messages:
        for msg in st.session_state.messages:
            role = "💬 You" if msg.get("role") == "user" else "🤖 MediBot"
            bubble_class = "user-bubble" if msg.get("role") == "user" else "assistant-bubble"
            content = msg.get("content", "")
            st.markdown(
                f"<div class='chat-bubble {bubble_class}'><strong>{role}:</strong> {escape(content)}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No messages yet. Start by typing your symptoms or uploading a report.")

# --- USER INPUT AND GENERATE REPLY ---
with control_col:
    st.markdown("### Actions")
    uploaded_file = st.file_uploader("Upload medical image/report (optional)", type=["jpg","jpeg","png","webp"])
    attachment_b64 = None
    if uploaded_file:
        file_bytes = uploaded_file.read()
        attachment_b64 = base64.b64encode(file_bytes).decode("utf-8")
        st.success("File uploaded and attached.")
        st.image(file_bytes, width=200)

st.markdown("---")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your medical question or symptoms...", key="user_input_text")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    user_msg = { "role": "user", "content": user_input.strip() }
    st.session_state.messages.append(user_msg)
    
    with st.spinner("MediBot is analyzing…"):
        reply_text = get_medibot_reply(st.session_state.messages, "English", attachment_b64)
        bot_msg = { "role": "assistant", "content": reply_text }
        st.session_state.messages.append(bot_msg)
    
    st.rerun()
