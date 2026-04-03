from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from rag.rrf import rrf_fusion
import pickle

# -----------------------------
# Load Stored Documents (for BM25)
# -----------------------------

def load_all_docs(path="docs.pkl"):
    print("[DIR] Loading documents for BM25...")
    with open(path, "rb") as f:
        docs = pickle.load(f)
    print(f"[OK] Loaded {len(docs)} documents")
    return docs

# -----------------------------
# Dense Retriever (FAISS)
# -----------------------------

def get_dense_retriever(vectordb, k=5):
    return vectordb.as_retriever(search_kwargs={"k": k})

# -----------------------------
# Sparse Retriever (BM25)
# -----------------------------

def get_sparse_retriever(all_docs, k=5):
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k
    return bm25

# -----------------------------
# Hybrid Retrieval
# -----------------------------

def hybrid_retrieval(query, vectordb, all_docs=None, docs_pkl_path="docs.pkl", k=10):
    """
    Combines Dense + Sparse retrieval using RRF fusion natively
    """
    print("\n[HYBRID] Running Hybrid Search...")

    # Load docs for BM25
    if all_docs is None:
        all_docs = load_all_docs(docs_pkl_path)

    # Create retrievers (over-fetch slightly for better RRF sorting)
    dense_retriever = get_dense_retriever(vectordb, k)
    sparse_retriever = get_sparse_retriever(all_docs, k)

    # Retrieve
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)

    print(f"Dense docs returned: {len(dense_docs)}")
    print(f"Sparse docs returned: {len(sparse_docs)}")

    # Use existing RRF fusion to merge results intelligently
    fused_docs = rrf_fusion([dense_docs, sparse_docs])
    
    print(f"[OK] Hybrid RRF final results: {len(fused_docs)}")

    return fused_docs