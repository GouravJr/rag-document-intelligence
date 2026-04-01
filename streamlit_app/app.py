"""
Streamlit Frontend — Intelligent Document Processor

Features:
  - Drag & drop PDF/image upload
  - Real-time OCR status with page-by-page feedback
  - Chat interface for document Q&A
  - Structured data extraction panel
  - Multi-document support
"""

import streamlit as st
import requests
import json
from pathlib import Path

# ── Config ──
API_URL = "http://localhost:8000"

# ── Page Config ──
st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-top: -10px;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .source-box {
        background: #f0f9ff;
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    .extraction-tag {
        display: inline-block;
        background: #ecfdf5;
        color: #065f46;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "extraction_data" not in st.session_state:
    st.session_state.extraction_data = {}


def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return None


# ── Sidebar ──
with st.sidebar:
    st.markdown("### 📄 RAG Document Intelligence")
    st.markdown("*AI-powered document analysis*")
    st.markdown("---")

    # Health check
    health = check_api_health()
    if health:
        st.success(f"✅ Backend connected — {health['documents_loaded']} doc(s) loaded")
    else:
        st.error("❌ Backend not running. Start it with:\n```\nuvicorn app.main:app --reload\n```")
        st.stop()

    st.markdown("---")
    st.markdown("### Upload documents")
    st.markdown("Supports PDF, PNG, JPG, TIFF")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Check if already processed
            if uploaded_file.name in [d.get("filename") for d in st.session_state.documents]:
                continue

            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/upload", files=files, timeout=120)

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.documents.append(data)

                        # Fetch extraction data
                        ext_response = requests.get(f"{API_URL}/extract/{data['doc_id']}", timeout=10)
                        if ext_response.status_code == 200:
                            st.session_state.extraction_data[data["doc_id"]] = ext_response.json()

                        st.success(f"✅ {uploaded_file.name}: {data['pages']} page(s), {data['chunks']} chunks")
                    else:
                        st.error(f"❌ {uploaded_file.name}: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name}: {str(e)}")

    # Document list
    if st.session_state.documents:
        st.markdown("---")
        st.markdown("### Loaded documents")
        for doc in st.session_state.documents:
            method_emoji = "🔍" if doc["method"] == "ocr" else "📝"
            st.markdown(
                f"{method_emoji} **{doc['filename']}**  \n"
                f"  {doc['pages']} pages · {doc['chunks']} chunks · {doc['method']}"
            )

        if st.button("🗑️ Clear all documents"):
            requests.delete(f"{API_URL}/clear", timeout=10)
            st.session_state.documents = []
            st.session_state.messages = []
            st.session_state.extraction_data = {}
            st.rerun()

    st.markdown("---")
    st.markdown(
        "Built with LangChain · FAISS · Groq  \n"
        "[GitHub](https://github.com/GouravJr) · Gourav Srinivasalu"
    )


# ── Main Content ──
st.markdown('<p class="main-header">📄 RAG Document Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload PDFs or images → OCR extracts text → Ask questions with RAG → Get structured data</p>', unsafe_allow_html=True)

# Metrics row
if st.session_state.documents:
    cols = st.columns(4)
    total_pages = sum(d["pages"] for d in st.session_state.documents)
    total_chunks = sum(d["chunks"] for d in st.session_state.documents)
    total_chars = sum(d["total_chars"] for d in st.session_state.documents)
    ocr_docs = sum(1 for d in st.session_state.documents if d["method"] in ("ocr", "hybrid"))

    cols[0].metric("Documents", len(st.session_state.documents))
    cols[1].metric("Pages", total_pages)
    cols[2].metric("Text chunks", total_chunks)
    cols[3].metric("OCR processed", ocr_docs)

# Tabs
if st.session_state.documents:
    tab_chat, tab_extract = st.tabs(["💬 Ask questions", "📊 Extracted data"])

    # ── Chat Tab ──
    with tab_chat:
        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 Sources"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="source-box">{src["content"]}</div>',
                                unsafe_allow_html=True,
                            )

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Show user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/query",
                            json={"question": prompt},
                            timeout=30,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.markdown(data["answer"])

                            if data.get("sources"):
                                with st.expander("📚 Sources"):
                                    for src in data["sources"]:
                                        st.markdown(
                                            f'<div class="source-box">{src["content"]}</div>',
                                            unsafe_allow_html=True,
                                        )

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": data["answer"],
                                "sources": data.get("sources", []),
                            })
                        else:
                            error = response.json().get("detail", "Unknown error")
                            st.error(f"Error: {error}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

    # ── Extraction Tab ──
    with tab_extract:
        for doc in st.session_state.documents:
            doc_id = doc["doc_id"]
            ext = st.session_state.extraction_data.get(doc_id, {})

            if not ext:
                continue

            st.markdown(f"#### 📄 {doc['filename']}")

            fields = ext.get("structured_fields", {})

            # Show extracted fields
            col1, col2 = st.columns(2)

            with col1:
                if fields.get("dates"):
                    st.markdown("**📅 Dates found:**")
                    for d in fields["dates"]:
                        st.markdown(f'<span class="extraction-tag">{d}</span>', unsafe_allow_html=True)

                if fields.get("amounts"):
                    st.markdown("**💰 Amounts found:**")
                    for a in fields["amounts"]:
                        st.markdown(f'<span class="extraction-tag">{a}</span>', unsafe_allow_html=True)

                if fields.get("reference_numbers"):
                    st.markdown("**🔖 References:**")
                    for r in fields["reference_numbers"]:
                        st.markdown(f'<span class="extraction-tag">{r}</span>', unsafe_allow_html=True)

            with col2:
                if fields.get("emails"):
                    st.markdown("**📧 Emails:**")
                    for e in fields["emails"]:
                        st.markdown(f'<span class="extraction-tag">{e}</span>', unsafe_allow_html=True)

                if fields.get("phones"):
                    st.markdown("**📞 Phone numbers:**")
                    for p in fields["phones"]:
                        st.markdown(f'<span class="extraction-tag">{p}</span>', unsafe_allow_html=True)

                if fields.get("ibans"):
                    st.markdown("**🏦 IBANs:**")
                    for iban in fields["ibans"]:
                        st.markdown(f'<span class="extraction-tag">{iban}</span>', unsafe_allow_html=True)

            # Key-Value pairs
            kv_pairs = ext.get("key_value_pairs", [])
            if kv_pairs:
                st.markdown("**📋 Key-Value pairs detected:**")
                for kv in kv_pairs[:20]:
                    st.markdown(f"- **{kv['key']}:** {kv['value']}")

            # Summary
            summary = fields.get("summary", {})
            if summary:
                lang = summary.get("document_language", "unknown")
                lang_label = {"de": "🇩🇪 German", "en": "🇬🇧 English"}.get(lang, "Unknown")
                st.info(
                    f"**Summary:** {summary.get('total_fields_found', 0)} fields extracted · "
                    f"Language: {lang_label} · "
                    f"Financial data: {'Yes' if summary.get('has_financial_data') else 'No'}"
                )

            st.markdown("---")

else:
    # Empty state
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            ### 👈 Upload a document to get started

            **What this tool does:**

            1. **OCR** — Extracts text from PDFs and scanned images (English + German)
            2. **RAG** — Lets you ask questions and get AI-powered answers with source citations
            3. **Extraction** — Automatically finds dates, amounts, emails, IBANs, and references

            **Try it with:**
            - German university enrollment forms
            - Invoices or contracts
            - Any PDF or scanned document
            """
        )
