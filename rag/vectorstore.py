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
        print("[LOAD] Initializing Local FastEmbed...", flush=True)
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        _embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=1)
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
    batch_size = 16
    total = len(documents)

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        print(f"   Embedding batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}...", flush=True)
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
    
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
