# 📄 RAG Document Intelligence

**Production-grade RAG + OCR pipeline for intelligent document processing.** Upload PDFs or scanned images, ask questions in natural language, and extract structured data — all running locally with zero cost.

Built for processing German and English documents: invoices, contracts, university forms, bureaucratic paperwork.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What it does

| Feature | Description |
|---------|-------------|
| **OCR** | Extracts text from native PDFs and scanned documents (Tesseract, English + German) |
| **RAG Q&A** | Ask questions about your documents — answers grounded in actual content with source citations |
| **Structured Extraction** | Automatically finds dates, amounts (EUR/USD), emails, phone numbers, IBANs, invoice numbers |
| **Multi-document** | Upload multiple documents and query across all of them |
| **Bilingual** | Handles both German and English documents natively |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                      │
│              Upload · Chat · Extraction View               │
└──────────────┬───────────────────────────┬───────────────┘
               │ REST API                   │
┌──────────────▼───────────────────────────▼───────────────┐
│                    FastAPI Backend                         │
│                                                           │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │   OCR   │   │  RAG Pipeline │   │   Structured     │  │
│  │         │   │              │   │   Extraction      │  │
│  │ PDF     │   │ Chunking    │   │                    │  │
│  │ native  │──▶│ Embedding   │   │ Dates · Amounts   │  │
│  │ +       │   │ FAISS Store │   │ Emails · IBANs    │  │
│  │ Tesseract│  │ LLM Query  │   │ References         │  │
│  └─────────┘   └──────┬───────┘   └──────────────────┘  │
│                        │                                  │
│              ┌─────────▼─────────┐                       │
│              │  Groq API (Free)  │                       │
│              │  Llama 3.1 8B     │                       │
│              └───────────────────┘                       │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | Groq (Llama 3.1 8B) | Free API, fast inference, no GPU needed |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Runs locally, no API key, 384-dim vectors |
| **Vector Store** | FAISS | Facebook's similarity search, in-memory, fast |
| **Framework** | LangChain | Industry-standard RAG orchestration |
| **OCR** | Tesseract + pdfplumber | Open-source, supports 100+ languages |
| **Backend** | FastAPI | Async, auto-docs at /docs, production-ready |
| **Frontend** | Streamlit | Rapid prototyping, built-in chat components |
| **Infra** | Docker + docker-compose | One-command deployment |

---

## Quick Start

### Option 1: Docker (recommended)

```bash
# Clone
git clone https://github.com/GouravJr/rag-document-intelligence.git
cd rag-document-intelligence

# Configure
cp .env.example .env
# Edit .env → add your GROQ_API_KEY (free at https://console.groq.com)

# Run
docker-compose up --build

# Open
# API docs: http://localhost:8000/docs
# Frontend: http://localhost:8501
```

### Option 2: Local Development

```bash
# Prerequisites: Python 3.11+, Tesseract OCR
# macOS: brew install tesseract tesseract-lang poppler
# Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-deu poppler-utils
# Windows: download from https://github.com/UB-Mannheim/tesseract/wiki

# Clone & setup
git clone https://github.com/GouravJr/rag-document-intelligence.git
cd rag-document-intelligence
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env → add your GROQ_API_KEY

# Terminal 1: Start API
uvicorn app.main:app --reload

# Terminal 2: Start Frontend
streamlit run streamlit_app/app.py

# Open http://localhost:8501
```

### Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Create an API key
4. Paste it in your `.env` file

---

## API Reference

Once running, full interactive docs at `http://localhost:8000/docs`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload PDF/image → OCR → ingest into RAG |
| `/query` | POST | Ask a question about uploaded documents |
| `/documents` | GET | List all ingested documents |
| `/extract/{doc_id}` | GET | Get structured data extraction |
| `/clear` | DELETE | Reset everything |
| `/health` | GET | Health check |

---

## Project Structure

```
rag-document-intelligence/
├── app/
│   ├── config.py          # Configuration & env vars
│   ├── main.py            # FastAPI endpoints
│   ├── ocr.py             # PDF & image text extraction
│   ├── rag.py             # RAG pipeline (chunk → embed → retrieve → generate)
│   └── extraction.py      # Structured data extraction (regex + heuristics)
├── streamlit_app/
│   └── app.py             # Streamlit chat + upload interface
├── tests/
│   └── test_pipeline.py   # Unit tests for extraction & pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## How it works

### 1. Document Ingestion
- Native PDFs: text extracted directly with `pdfplumber` (fast, accurate)
- Scanned PDFs/images: pages converted to images → Tesseract OCR (300 DPI, preprocessed)
- Hybrid mode: automatically detects which pages need OCR

### 2. RAG Pipeline
- Text split into overlapping chunks (1000 chars, 200 overlap)
- Chunks embedded with `sentence-transformers/all-MiniLM-L6-v2` (runs on CPU)
- Stored in FAISS index for sub-millisecond similarity search
- On query: top-4 relevant chunks retrieved → sent to Groq LLM with custom prompt → grounded answer returned

### 3. Structured Extraction
- Regex-based extraction of dates, amounts, emails, phones, IBANs, reference numbers
- Supports German date formats (DD.MM.YYYY) and currency notation (1.234,56 EUR)
- Key-value pair detection for form-like documents
- Automatic language detection (German vs English)

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Future Improvements

- [ ] Add table extraction from PDFs (with `camelot` or `tabula`)
- [ ] Fine-tune embedding model on German documents
- [ ] Add authentication for multi-user deployment
- [ ] Implement conversation memory for follow-up questions
- [ ] Add Kubernetes deployment manifests
- [ ] Support for XLSX and DOCX input

---

## Author

**Gourav Srinivasalu**
M.Sc. Artificial Intelligence Engineering · Universität Passau

- GitHub: [github.com/GouravJr](https://github.com/GouravJr)
- LinkedIn: [linkedin.com/in/gourav-srinivasalu](https://linkedin.com/in/gourav-srinivasalu)
- Email: gourav.srinivasalu@gmail.com

---

## License

MIT License — see [LICENSE](LICENSE) for details.
