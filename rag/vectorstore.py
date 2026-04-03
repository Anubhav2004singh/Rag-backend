import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "db/faiss_indexes"
os.makedirs(DB_DIR, exist_ok=True)

# -----------------------------
# Lazy-loaded Embedding Model
# -----------------------------

_embedding_model = None

def get_embedding():
    global _embedding_model
    if _embedding_model is None:
        print("[LOAD] Initializing Local HuggingFace Embeddings...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


# -----------------------------
# Create & Store Vector DB
# -----------------------------

def create_vector_store(documents, collection_name="default"):
    """
    Create vector store for documents
    """
    print(f"Creating embeddings and storing in FAISS (collection: {collection_name})...")

    embeddings = get_embedding()
    
    # FAISS creation is extremely fast and entirely in memory natively
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the index locally to disk
    save_path = os.path.join(DB_DIR, collection_name)
    vectorstore.save_local(save_path)

    print("[OK] Vector store saved at:", save_path)
    return vectorstore


# -----------------------------
# Load Existing Vector DB
# -----------------------------

def load_vectorstore(collection_name="default"):
    """
    Load existing vector store
    """
    print(f"Loading existing vector store (collection: {collection_name})...")

    embeddings = get_embedding()
    save_path = os.path.join(DB_DIR, collection_name)

    vectorstore = FAISS.load_local(
        save_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    print("[OK] Vector store loaded")
    return vectorstore
