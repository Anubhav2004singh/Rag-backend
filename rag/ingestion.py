import json
from typing import List
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================
# FAST PDF Text Extraction (using pdfplumber)
# =============================================

def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """
    Extract text from PDF using pdfplumber (FAST).
    Returns (full_text, page_count)
    """
    import pdfplumber

    print(f"[FILE] Extracting text from: {file_path}")

    full_text = ""
    page_count = 0

    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)
        print(f"   Pages: {page_count}")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            # Also extract tables as text
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                for row in table:
                    if row:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        table_text += " | ".join(cleaned) + "\n"

            page_text = text
            if table_text:
                page_text += f"\n\n[Table from page {i+1}]\n{table_text}"

            full_text += page_text + "\n\n"

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
        print("[WARN] No text extracted from PDF. It might be a scanned/image-only PDF.")
        print("   Falling back to basic extraction...")
        # Try with a simpler approach for scanned PDFs
        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(filename=pdf_path, strategy="fast")
            text = "\n\n".join([el.text for el in elements if el.text])
        except Exception as e:
            print(f"   Fallback also failed: {e}")
            return []

    # Step 2: Chunk
    documents = chunk_text(text)

    # Add page count metadata
    for doc in documents:
        doc.metadata["source_pages"] = page_count

    print(f"\n[OK] Pipeline complete: {len(documents)} chunks from {page_count} pages")
    return documents
