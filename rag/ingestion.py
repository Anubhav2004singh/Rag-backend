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
    import platform
    import os

    print(f"[FILE] Extracting text from: {file_path}", flush=True)

    full_text = ""
    page_count = 0

    try:
        with fitz.open(file_path) as pdf:
            page_count = len(pdf)
            print(f"   Pages: {page_count}", flush=True)

            for page in pdf:
                page_text = page.get_text()
                
                # 1. Independent Embedded Image Extraction
                # Process specific pictures within textual pages without double-reading native text!
                image_list = page.get_images()
                if image_list:
                    try:
                        import pytesseract
                        from PIL import Image
                        import io
                        
                        if platform.system() == "Windows":
                            default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                            if os.path.exists(default_path):
                                pytesseract.pytesseract.tesseract_cmd = default_path
                                
                        for img_index, img in enumerate(image_list):
                            xref = img[0]
                            base_image = pdf.extract_image(xref)
                            if base_image and "image" in base_image:
                                img_pil = Image.open(io.BytesIO(base_image["image"]))
                                # Normalize image modes to prevent Tesseract crashes
                                if img_pil.mode not in ("L", "RGB", "RGBA"):
                                    img_pil = img_pil.convert("RGB")
                                ocr_text = pytesseract.image_to_string(img_pil)
                                if ocr_text.strip():
                                    page_text += f"\n[Embedded Image {img_index+1} Content]: {ocr_text.strip()}\n"

                    except Exception as ocr_err:
                        print(f"      [WARN] Embedded Image OCR Failed: {ocr_err}", flush=True)

                
                # 2. Master Fallback: If page is STILL totally blank, rasterize the entire physical page 
                if not page_text.strip():
                    try:
                        import pytesseract
                        from PIL import Image
                        
                        if platform.system() == "Windows":
                            default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                            if os.path.exists(default_path):
                                pytesseract.pytesseract.tesseract_cmd = default_path
                                
                        print(f"      [OCR] Blank page detected. Booting PyTesseract Full-Page Neural vision...", flush=True)
                        pix = page.get_pixmap(dpi=150) # Moderate DPI for speed vs accuracy balance
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        ocr_text = pytesseract.image_to_string(img)
                        page_text += ocr_text + "\n"
                        
                    except Exception as ocr_err:
                        print(f"      [WARN] Full-Page OCR Failed: {ocr_err}", flush=True)
                
                full_text += page_text + "\n\n"
                
    except Exception as e:
        print(f"Error extracting PDF: {e}", flush=True)

    print(f"   Extracted {len(full_text)} characters", flush=True)
    return full_text.strip(), page_count


# =============================================
# Fast Chunking (LangChain text splitter)
# =============================================

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    Much faster than unstructured's chunk_by_title.
    """
    print(f"[CUT] Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.create_documents([text])

    print(f"   Created {len(chunks)} chunks", flush=True)
    return chunks


# =============================================
# Omni-Format Router
# =============================================

def extract_text_from_image(file_path: str) -> tuple[str, int]:
    """Bypasses PDF logic and natively pipes standard Images (JPG/PNG) straight into Tesseract."""
    print(f"[FILE] Extracting text directly from Image: {file_path}", flush=True)
    try:
        import pytesseract
        from PIL import Image
        import platform
        if platform.system() == "Windows":
            default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_path):
                pytesseract.pytesseract.tesseract_cmd = default_path
        
        img = Image.open(file_path)
        if img.mode not in ("L", "RGB", "RGBA"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip(), 1
    except Exception as e:
        print(f"      [WARN] Image OCR Failed: {e}", flush=True)
        return "", 1

def extract_text_from_txt(file_path: str) -> tuple[str, int]:
    """Instantly digests raw strings out of TXT, CSV, MD, or JSON wrappers."""
    print(f"[FILE] Extracting string locally from Flat Text File: {file_path}", flush=True)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text.strip(), 1
    except Exception as e:
        print(f"      [WARN] Flat Text Reading failed: {e}", flush=True)
        return "", 1

def omni_extract(file_path: str) -> tuple[str, int]:
    ext = file_path.lower().split('.')[-1]
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['jpg', 'jpeg', 'png', 'webp', 'bmp']:
        return extract_text_from_image(file_path)
    elif ext in ['txt', 'md', 'csv', 'json']:
        return extract_text_from_txt(file_path)
    else:
        # Grand master fallback
        print(f"[WARN] Unknown extension gracefully falling back to flat text read: {ext}", flush=True)
        return extract_text_from_txt(file_path)


# =============================================
# Full Pipeline
# =============================================

def run_complete_ingestion_pipeline(file_path: str) -> List[Document]:
    """
    Fast Omni-ingestion pipeline:
    1. Determine format and rip native text (or OCR vision) natively.
    2. Chunk with LangChain vector structures.
    3. Push directly to embeddings.
    """
    print("=" * 50, flush=True)
    print(f"[>>] Starting OMNI ingestion on: {file_path}", flush=True)
    print("=" * 50, flush=True)

    # Step 1: Extract text dynamically
    text, page_count = omni_extract(file_path)

    if not text.strip():
        print("[FAIL] OCR completely failed to scan ANY text from this file.", flush=True)
        return []

    # Step 2: Chunk
    documents = chunk_text(text)

    # Add page count metadata
    for doc in documents:
        doc.metadata["source_pages"] = page_count

    print(f"\n[OK] Pipeline complete: {len(documents)} chunks from {page_count} pages", flush=True)
    return documents
