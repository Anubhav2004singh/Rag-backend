import json
from typing import List
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================
# LIGHTNING PDF Text Extraction (using PyMuPDF)
# =============================================

def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """
    Extract text from PDF using PyMuPDF (fitz) for extreme speed.
    Returns (full_text, page_count)
    """
    import fitz  # PyMuPDF

    print(f"[FILE] Extracting text from: {file_path}")

    full_text = ""
    page_count = 0

    try:
        with fitz.open(file_path) as pdf:
            page_count = len(pdf)
            print(f"   Pages: {page_count}")

            for page in pdf:
                full_text += page.get_text() + "\n\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")

    print(f"   Extracted {len(full_text)} characters")
    return full_text.strip(), page_count


# =============================================
# Fast Chunking (LangChain text splitter)
# =============================================

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[Document]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    Much faster than unstructured's chunk_by_title.
    """
    print(f"[CUT] Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.create_documents([text])

    print(f"   Created {len(chunks)} chunks")
    return chunks


# =============================================
# Full Pipeline
# =============================================

def run_complete_ingestion_pipeline(pdf_path: str) -> List[Document]:
    """
    Fast ingestion pipeline:
    1. Extract text with pdfplumber (fast)
    2. Chunk with RecursiveCharacterTextSplitter (fast)
    3. Return LangChain Documents ready for embedding

    Typically completes in < 30 seconds for most PDFs.
    """
    print("=" * 50)
    print("[>>] Starting FAST ingestion pipeline")
    print("=" * 50)

    # Step 1: Extract text
    text, page_count = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print("[WARN] No text extracted from PDF (likely an image). OCR is currently disabled for max speed.")
        return []

    # Step 2: Chunk
    documents = chunk_text(text)

    # Add page count metadata
    for doc in documents:
        doc.metadata["source_pages"] = page_count

    print(f"\n[OK] Pipeline complete: {len(documents)} chunks from {page_count} pages")
    return documents
