import json
from typing import List
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================
# Google Gemini 1.5 Flash Vision OCR Adapter
# =============================================

def google_vision_ocr(image) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    import base64
    from io import BytesIO

    # Serialize PIL Image into base64 bypass
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Extract all the text inside this image perfectly. Maintain the exact formatting. Do not add any conversational text or prefix, just output the exact text you see. If there is absolutely no text, return an empty string.",
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}},
        ]
    )
    
    response = llm.invoke([message])
    return response.content

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
                        from PIL import Image
                        import io
                        
                        for img_index, img in enumerate(image_list):
                            xref = img[0]
                            base_image = pdf.extract_image(xref)
                            if base_image and "image" in base_image:
                                img_pil = Image.open(io.BytesIO(base_image["image"]))
                                # Normalize image modes
                                if img_pil.mode not in ("L", "RGB", "RGBA"):
                                    img_pil = img_pil.convert("RGB")
                                
                                ocr_text = google_vision_ocr(img_pil)
                                
                                if ocr_text.strip():
                                    page_text += f"\n[Embedded Image {img_index+1} Content]: {ocr_text.strip()}\n"

                    except Exception as ocr_err:
                        print(f"      [WARN] Embedded Image OCR Failed: {ocr_err}", flush=True)

                
                # 2. Master Fallback: If page is STILL totally blank, rasterize the entire physical page 
                if not page_text.strip():
                    try:
                        from PIL import Image
                        
                        print(f"      [OCR] Blank page detected. Booting Gemini 1.5 Flash Vision OCR...", flush=True)
                        pix = page.get_pixmap(dpi=150) # Moderate DPI for speed vs accuracy balance
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        ocr_text = google_vision_ocr(img)
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
    """Bypasses PDF logic and natively pipes standard Images (JPG/PNG) straight into Gemini Vision."""
    print(f"[FILE] Extracting text directly from Image: {file_path}", flush=True)
    try:
        from PIL import Image
        
        img = Image.open(file_path)
        if img.mode not in ("L", "RGB", "RGBA"):
            img = img.convert("RGB")
            
        print(f"      [OCR] Booting Gemini 1.5 Flash Vision OCR...", flush=True)
        text = google_vision_ocr(img)
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
