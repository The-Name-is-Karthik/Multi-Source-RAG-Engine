"""
Main Streamlit application for the Multi-Source RAG Engine.

This version includes:
- A complete visual overhaul with a professional, modern UI.
- Streaming responses for an interactive feel.
- Caching for all expensive operations for speed.
- AI-generated suggested questions to guide the user.
- Robust resource management with no memory leaks.
- Proper state management to prevent data source contamination.
- A responsive UI where all interactive elements trigger the AI correctly.


"""

import streamlit as st
import os
import re
import hashlib
import logging
from langchain.schema.document import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from src.video_processor import get_video_transcript
from src.data_loader import load_from_webpage, load_from_pdf, load_from_docx
from src.vector_store import create_vector_store
from src.rag_pipeline import create_rag_chain

# --- Page Configuration ---
st.set_page_config(page_title="Multi-Source RAG Engine", page_icon="🌀", layout="wide")

# --- Load Custom CSS ---
@st.cache_data
def load_css(file_path: str):
    with open(file_path) as f:
        return f.read()

st.markdown(f"<style>{load_css('styles/style.css')}</style>", unsafe_allow_html=True)


# --- App State Management ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "source_name" not in st.session_state:
    st.session_state.source_name = ""

# --- Helper Functions ---
@st.cache_data
def generate_suggested_questions(_docs: list[Document]):
    if not _docs: return []
    combined_content = " ".join([doc.page_content for doc in _docs[:3]])[:4000]
    prompt = f'''Based on the following text, generate 3 concise, insightful questions a user might want to ask. The questions should be distinct.\n\nText:\n\"""{combined_content}\"""\n\nQuestions:'''
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GEMINI_API_KEY)
        response = llm.invoke(prompt)
        questions = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', response.content, re.DOTALL)
        return [q.strip() for q in questions if q.strip()]
    except Exception as e:
        logging.error(f"Failed to generate suggested questions: {e}")
        return []

@st.cache_data
def load_and_cache_document(file_content: bytes, file_name: str, file_type: str) -> list[Document]:
    file_hash = hashlib.md5(file_content).hexdigest()
    temp_dir = "temp_docs_cache"
    os.makedirs(temp_dir, exist_ok=True)
    file_extension = os.path.splitext(file_name)[1]
    temp_path = os.path.join(temp_dir, f"{file_hash}{file_extension}")
    with open(temp_path, "wb") as f:
        f.write(file_content)
    try:
        if file_type == "application/pdf": return load_from_pdf(temp_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": return load_from_docx(temp_path)
        return []
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

def process_new_source(source_docs: list[Document], source_name: str):
    st.cache_resource.clear()
    st.cache_data.clear()
    vector_store = create_vector_store(source_docs)
    st.session_state.rag_chain = create_rag_chain(vector_store.as_retriever())
    st.session_state.chat_history = []
    st.session_state.source_name = source_name
    st.session_state.suggested_questions = generate_suggested_questions(source_docs)
    st.toast(f"Ready to chat with {source_name}!", icon="✅")

# --- Header ---
st.markdown("<h1 style='text-align:center; margin-bottom:0;'>🌀 Multi Source RAG Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.2rem; color:var(--text-color);'>Your Intelligent Assistant for Documents and Media Insights</p>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📂 Data Sources")
    url = st.text_input("🔗 URL (YouTube, Web Page)", key="url_input", placeholder="https://...")
    if st.button("🚀 Process URL"):
        if url:
            with st.spinner("Processing URL..."):
                docs = []
                if "youtube.com" in url or "youtu.be" in url:
                    transcript = get_video_transcript(url)
                    if transcript: docs = [Document(page_content=transcript)]
                else:
                    docs = load_from_webpage(url)
                if docs: process_new_source(docs, url); st.rerun()
                else: st.error("Could not retrieve content.")
        else: st.warning("Please enter a URL.")

    uploaded_files = st.file_uploader("📄 Upload Docs (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("📥 Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_docs = []
                source_name = ", ".join([f.name for f in uploaded_files])
                for up_file in uploaded_files:
                    file_content = up_file.getvalue()
                    all_docs.extend(load_and_cache_document(file_content, up_file.name, up_file.type))
                if all_docs: process_new_source(all_docs, source_name); st.rerun()
                else: st.error("Could not extract content.")
        else: st.warning("Please upload at least one document.")

# --- Main Chat Interface ---
if not st.session_state.rag_chain:
    st.info("Please process a data source from the sidebar to start a conversation.")
else:
    st.markdown(f"**Chatting with:** `{st.session_state.source_name}`")

    # Chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            st.markdown(f"<div class='chat-bubble chat-assistant'>{msg.content}</div>", unsafe_allow_html=True)
            if msg.content.startswith("Based on the provided context:") and msg.additional_kwargs.get("context"):
                with st.expander("📚 Show Sources"):
                    for doc in msg.additional_kwargs["context"]:
                        st.info(doc.page_content)
        elif isinstance(msg, HumanMessage):
            st.markdown(f"<div class='chat-bubble chat-user'>{msg.content}</div>", unsafe_allow_html=True)

    # Response streaming
    if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
        prompt = st.session_state.chat_history[-1].content
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            full_response = ""
            context_docs = []
            response_stream = st.session_state.rag_chain.stream({"input": prompt, "chat_history": st.session_state.chat_history[:-1]})
            for chunk in response_stream:
                if answer_chunk := chunk.get("answer"): full_response += answer_chunk; placeholder.markdown(full_response + "▌")
                if context_chunk := chunk.get("context"): context_docs = context_chunk
            placeholder.markdown(full_response)
        ai_msg_with_context = AIMessage(content=full_response, additional_kwargs={"context": context_docs})
        st.session_state.chat_history.append(ai_msg_with_context)
        st.session_state.suggested_questions = []
        st.rerun()

    # Suggested Questions
    if st.session_state.suggested_questions:
        st.markdown("### 💡 Suggested Questions")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, q in enumerate(st.session_state.suggested_questions):
            if cols[i].button(q, key=f"suggestion_{i}"):
                st.session_state.chat_history.append(HumanMessage(content=q)); st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append(HumanMessage(content=prompt)); st.rerun()
