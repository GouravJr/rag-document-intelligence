"""
RAG Module — Retrieval-Augmented Generation pipeline.

Pipeline:
  1. Text → Chunks (RecursiveCharacterTextSplitter)
  2. Chunks → Embeddings (sentence-transformers, runs locally)
  3. Embeddings → FAISS vector store (no external DB needed)
  4. Query → Retrieve top-k chunks → LLM generates answer with context

Uses Groq API (free tier) for LLM inference — no GPU required.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    VECTOR_STORE_DIR,
)

logger = logging.getLogger(__name__)


# ── Prompt Template ──
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an intelligent document assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I cannot find this information in the uploaded documents."

Be precise, cite specific details from the document, and structure your answer clearly.

Context:
{context}

Question: {question}

Answer:"""
)


class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest documents, build vector store, answer queries.

    Usage:
        rag = RAGPipeline()
        doc_id = rag.ingest("extracted text from OCR...", metadata={...})
        answer = rag.query("What is the total amount on this invoice?")
    """

    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self._vector_store: Optional[FAISS] = None
        self._qa_chain = None
        self._documents: dict[str, dict] = {}  # doc_id -> metadata

    @property
    def is_ready(self) -> bool:
        return self._vector_store is not None

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def ingest(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        Ingest a document into the vector store.

        Args:
            text: Full document text (from OCR module)
            metadata: Optional metadata (filename, pages, etc.)

        Returns:
            doc_id: Unique identifier for this document
        """
        doc_id = hashlib.md5(text[:500].encode()).hexdigest()[:12]
        meta = metadata or {}
        meta["doc_id"] = doc_id

        # Split into chunks
        chunks = self._splitter.create_documents(
            texts=[text],
            metadatas=[meta],
        )
        logger.info(f"Document {doc_id}: {len(text)} chars → {len(chunks)} chunks")

        # Build or extend vector store
        if self._vector_store is None:
            self._vector_store = FAISS.from_documents(chunks, self._embeddings)
        else:
            self._vector_store.add_documents(chunks)

        # Rebuild QA chain with updated store
        self._build_qa_chain()

        # Track document
        self._documents[doc_id] = {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "chars": len(text),
            **meta,
        }

        return doc_id

    def query(self, question: str) -> dict:
        """
        Ask a question about the ingested documents.

        Returns:
            dict with keys: answer, sources (list of relevant chunks)
        """
        if not self.is_ready:
            return {
                "answer": "No documents have been uploaded yet. Please upload a document first.",
                "sources": [],
            }

        # Retrieve relevant chunks
        retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS},
        )
        relevant_docs = retriever.invoke(question)

        # Generate answer via LLM
        result = self._qa_chain.invoke({"query": question})

        sources = [
            {
                "content": doc.page_content[:300],
                "metadata": doc.metadata,
            }
            for doc in relevant_docs
        ]

        return {
            "answer": result["result"],
            "sources": sources,
            "model": LLM_MODEL,
        }

    def get_similar_chunks(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k most similar chunks without LLM generation."""
        if not self.is_ready:
            return []

        docs = self._vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "score": round(float(score), 4),
                "metadata": doc.metadata,
            }
            for doc, score in docs
        ]

    def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        return list(self._documents.values())

    def clear(self):
        """Clear all documents and reset the pipeline."""
        self._vector_store = None
        self._qa_chain = None
        self._documents.clear()
        logger.info("RAG pipeline cleared")

    def _build_qa_chain(self):
        """Build the RetrievalQA chain with the current vector store."""
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set — query() will fail")
            return

        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=1024,
        )

        self._qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self._vector_store.as_retriever(
                search_kwargs={"k": TOP_K_RESULTS}
            ),
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=False,
        )
