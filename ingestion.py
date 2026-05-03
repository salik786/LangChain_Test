

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# ── tuneable constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 500          # approximate token budget per chunk
CHUNK_OVERLAP = 50        # token overlap between adjacent chunks
RETRIEVER_K = 6           # number of chunks returned per query
# SCORE_THRESHOLD = 0.30    # minimum similarity score (0–1); below → "not found"


def _load_pdf_text(pdf_path: str | Path) -> str:
    """Extract raw text from every page of the PDF using pypdf."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            logger.warning("Page %d: no text extracted (possibly image-only)", i + 1)
        pages.append(text)
    full_text = "\n\n".join(pages)
    logger.info("PDF loaded: %d pages, ~%d characters", len(pages), len(full_text))
    return full_text


def _rough_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple character-based sliding window chunker.
    Converts token budgets to character budgets (×4) then splits on
    sentence/paragraph boundaries where possible.
    """
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunk = text[start:end]
        # Try to break on a paragraph boundary within the last 20 % of the window
        search_start = max(start, end - char_size // 5)
        para_break = text.rfind("\n\n", search_start, end)
        if para_break != -1:
            chunk = text[start:para_break]
            end = para_break
        chunks.append(chunk.strip())
        start = end - char_overlap
        if start >= len(text):
            break

    logger.info("Split into %d chunks", len(chunks))
    return [c for c in chunks if c]


def build_retriever(pdf_path: str | Path) -> VectorStoreRetriever:
    """
    Main entry point called once on agent boot.

    Returns a LangChain VectorStoreRetriever backed by an in-memory FAISS index.

    Production note: swap FAISS for a managed vector store such as
    pgvector (via langchain-postgres) or Pinecone for persistence,
    horizontal scaling, and metadata filtering.
    """
    text = _load_pdf_text(pdf_path)
    raw_chunks = _split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(raw_chunks)
    ]

    # Lazy import so startup only imports the chosen LLM provider
    embeddings = _get_embeddings()

    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(
    search_kwargs={"k": RETRIEVER_K},
)
    logger.info("FAISS index built with %d documents", len(docs))
    return retriever


def _get_embeddings():
    """Auto-detect available provider from environment variables."""
    import os

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    if os.getenv("ANTHROPIC_API_KEY"):
        # Anthropic doesn't provide embeddings; fall back to a local model
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.warning(
            "Anthropic does not provide embeddings. "
            "Using local HuggingFace model (sentence-transformers/all-MiniLM-L6-v2). "
            "Install sentence-transformers if not already present."
        )
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        )

    raise EnvironmentError(
        "No LLM provider key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "or AZURE_OPENAI_API_KEY in your .env file."
    )
