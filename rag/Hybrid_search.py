from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv
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
# Dense Retriever (Chroma)
# -----------------------------

def get_dense_retriever(vectordb, k=5):

    return vectordb.as_retriever(
        search_kwargs={"k": k}
    )


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

def hybrid_retrieval(query, vectordb, all_docs=None, docs_pkl_path="docs.pkl", k=5):
    """
    Combines Dense + Sparse retrieval

    Args:
        query: user query
        vectordb: loaded vector database
        all_docs: pre-loaded documents for BM25 (optional, loads from pkl if None)
        docs_pkl_path: path to docs pickle file (used if all_docs is None)
        k: number of results per retriever
    """

    print("\n[HYBRID] Running Hybrid Search...")

    # Load docs for BM25
    if all_docs is None:
        all_docs = load_all_docs(docs_pkl_path)

    # Create retrievers
    dense_retriever = get_dense_retriever(vectordb, k)
    sparse_retriever = get_sparse_retriever(all_docs, k)

    # Retrieve
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)

    print(f"Dense docs: {len(dense_docs)}")
    print(f"Sparse docs: {len(sparse_docs)}")

    # Combine results
    combined_docs = dense_docs + sparse_docs

    # Remove duplicates
    unique_docs = []
    seen = set()

    for doc in combined_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    print(f"[OK] Hybrid results after deduplication: {len(unique_docs)}")

    return unique_docs