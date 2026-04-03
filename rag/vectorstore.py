import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

DB_DIR = "db/faiss_indexes"
os.makedirs(DB_DIR, exist_ok=True)

# -----------------------------
# Lazy-loaded Embedding Model
# -----------------------------

_embedding_model = None

def get_embedding():
    global _embedding_model
    if _embedding_model is None:
        print("[LOAD] Initializing Google API Embeddings (Zero RAM footprint)...", flush=True)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        # Using the ultra-stable gemini-embedding-001 model for compatibility 
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            task_type="retrieval_document"
        )
    return _embedding_model


# -----------------------------
# Create & Store Vector DB
# -----------------------------

def create_vector_store(documents, collection_name="default"):
    """
    Create vector store for documents
    """
    print(f"Creating embeddings and storing in FAISS (collection: {collection_name})...", flush=True)

    embeddings = get_embedding()
    
    vectorstore = None
    # Google's free tier is strictly 100 Requests Per Minute. Langchain's Google wrapper
    # sends items concurrently, causing instant 429 crashes on large batches. 
    # By strictly reducing batch_size to 2 and enforcing a time buffer, we mathematically bypass it.
    batch_size = 2
    total = len(documents)

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        print(f"   Embedding chunk batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} via Google API...", flush=True)
        import time
        
        max_retries = 6
        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break # Success!
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 20 # Google's quota resets typically every 15s or 1 minute
                        print(f"      [RATE LIMIT] Google API quota hit. Sleeping {wait_time}s (Attempt {attempt+1}/{max_retries})...", flush=True)
                        time.sleep(wait_time)
                    else:
                        raise e
                else:
                    raise e
        
        # Base delay to organically stretch out batches and prevent RPM exhaustion
        time.sleep(1.5)
    
    # Save the index locally to disk
    save_path = os.path.join(DB_DIR, collection_name)
    vectorstore.save_local(save_path)

    print("[OK] Vector store saved at:", save_path, flush=True)
    return vectorstore


# -----------------------------
# Load Existing Vector DB
# -----------------------------

def load_vectorstore(collection_name="default"):
    """
    Load existing vector store
    """
    print(f"Loading existing vector store (collection: {collection_name})...", flush=True)

    embeddings = get_embedding()
    save_path = os.path.join(DB_DIR, collection_name)

    vectorstore = FAISS.load_local(
        save_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    print("[OK] Vector store loaded", flush=True)
    return vectorstore


# -----------------------------
# Multi-Document Merge
# -----------------------------

def load_all_vectorstores():
    """
    Loads all FAISS instances in the DB_DIR and perfectly merges them 
    into an overarching global multidimensional vector space.
    """
    print("[MULTI] Preparing dynamically merged Global Vector Store...", flush=True)
    
    embeddings = get_embedding()
    base_vectorstore = None
    merged_count = 0

    if not os.path.exists(DB_DIR):
        print("[ERROR] No database directory found.", flush=True)
        return None

    # Iterate through all saved collections
    for collection_dir in os.listdir(DB_DIR):
        full_path = os.path.join(DB_DIR, collection_dir)
        if os.path.isdir(full_path) and collection_dir.startswith("doc_"):
            try:
                temp_vs = FAISS.load_local(
                    full_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                if base_vectorstore is None:
                    base_vectorstore = temp_vs
                else:
                    base_vectorstore.merge_from(temp_vs)
                merged_count += 1
            except Exception as e:
                print(f"[WARN] Failed to load or merge vectorstore {collection_dir}: {e}", flush=True)
                
    print(f"[OK] Globally merged {merged_count} FAISS Vector Stores.", flush=True)
    return base_vectorstore
