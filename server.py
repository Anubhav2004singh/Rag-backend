"""
RAG Backend - FastAPI Server (Render Ready)
"""

import os
import json
import uuid
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =============================================
# ENV
# =============================================

load_dotenv(override=True)

# =============================================
# PATHS
# =============================================

DOCS_DIR = Path("docs")
DB_DIR = Path("db")
DOCS_META_FILE = DB_DIR / "documents.json"
DOCS_PKL_DIR = DB_DIR / "docs_pkl"

DOCS_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
DOCS_PKL_DIR.mkdir(exist_ok=True)

# =============================================
# APP + CORS
# =============================================

app = FastAPI(title="Curator RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================
# MODELS
# =============================================

class ChatRequest(BaseModel):
    query: str
    document_id: str

# =============================================
# UTIL FUNCTIONS
# =============================================

def _load_documents_meta():
    if DOCS_META_FILE.exists():
        try:
            return json.load(open(DOCS_META_FILE))
        except Exception:
            return []
    return []

def _save_documents_meta(docs):
    with open(DOCS_META_FILE, "w") as f:
        json.dump(docs, f, indent=2)

def _update_document_status(doc_id, **updates):
    docs = _load_documents_meta()
    for d in docs:
        if d["id"] == doc_id:
            d.update(updates)
    _save_documents_meta(docs)

def _get_document_meta(doc_id):
    docs = _load_documents_meta()
    return next((d for d in docs if d["id"] == doc_id), None)

def _format_file_size(size_bytes):
    if size_bytes == 0:
        return "0 Bytes"
    import math
    sizes = ["Bytes", "KB", "MB", "GB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    return f"{round(size_bytes / (1024 ** i), 1)} {sizes[i]}"

# =============================================
# ROUTES
# =============================================

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Curator RAG API is running"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload PDF and process it SYNCHRONOUSLY.
    On Render free tier, background threads get killed when the service
    spins down, so we process inline. The fast pipeline takes ~10-30 seconds.
    """

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    doc_id = str(uuid.uuid4())[:8]
    file_path = DOCS_DIR / f"{doc_id}_{file.filename}"

    # Save file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Save initial metadata
    docs = _load_documents_meta()
    doc_meta = {
        "id": doc_id,
        "name": file.filename,
        "size": _format_file_size(len(contents)),
        "uploaded_at": datetime.now().isoformat(),
        "status": "indexing",
        "file_path": str(file_path),
        "error_message": None,
        "chunk_count": None,
    }
    docs.insert(0, doc_meta)
    _save_documents_meta(docs)

    # Process SYNCHRONOUSLY (no background thread)
    try:
        from rag.ingestion import run_complete_ingestion_pipeline
        from rag.vectorstore import create_vector_store

        print(flush=True, f"Processing {doc_id}: {file.filename}")
        chunk_docs = run_complete_ingestion_pipeline(str(file_path))

        if not chunk_docs:
            _update_document_status(doc_id, status="error",
                                    error_message="No text could be extracted from this PDF")
            return {"id": doc_id, "status": "error", "message": "No text extracted"}

        # Save docs pickle for BM25
        pkl_path = DOCS_PKL_DIR / f"{doc_id}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(chunk_docs, f)

        # Create vector store
        collection_name = f"doc_{doc_id.replace('-', '_')}"
        create_vector_store(chunk_docs, collection_name=collection_name)

        # Mark as processed
        _update_document_status(doc_id, status="processed", chunk_count=len(chunk_docs))
        print(flush=True, f"Done {doc_id}: {len(chunk_docs)} chunks")

        return {"id": doc_id, "status": "processed", "chunks": len(chunk_docs)}

    except Exception as e:
        traceback.print_exc()
        _update_document_status(doc_id, status="error", error_message=str(e)[:300])
        return {"id": doc_id, "status": "error", "message": str(e)[:300]}


@app.get("/api/documents")
async def list_docs():
    return _load_documents_meta()


@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: str):
    doc = _get_document_meta(doc_id)
    if not doc:
        raise HTTPException(404, "Not found")

    # Remove file
    fp = doc.get("file_path")
    if fp and os.path.exists(fp):
        os.remove(fp)

    # Remove pickle
    pkl = DOCS_PKL_DIR / f"{doc_id}.pkl"
    if pkl.exists():
        pkl.unlink()

    # Remove from metadata
    docs = [d for d in _load_documents_meta() if d["id"] != doc_id]
    _save_documents_meta(docs)

    return {"message": "Deleted", "id": doc_id}


@app.post("/api/chat")
async def chat(request: ChatRequest):

    doc = _get_document_meta(request.document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    if doc["status"] != "processed":
        raise HTTPException(400, "Document is still processing. Please wait.")

    try:
        from rag.vectorstore import load_vectorstore
        from rag.query_pipeline import query_rag

        collection_name = f"doc_{request.document_id.replace('-', '_')}"
        vectordb = load_vectorstore(collection_name)

        pkl_path = DOCS_PKL_DIR / f"{request.document_id}.pkl"
        all_docs = None
        if pkl_path.exists():
            with open(pkl_path, "rb") as pf:
                all_docs = pickle.load(pf)

        answer, source_docs = query_rag(
            request.query,
            vectordb,
            all_docs=all_docs
        )

        sources = [d.page_content[:200] for d in source_docs[:3]]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Error: {str(e)[:300]}")


# =============================================
# ENTRY POINT
# =============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(flush=True, f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
