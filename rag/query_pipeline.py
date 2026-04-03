from rag.Hybrid_search import hybrid_retrieval
from rag.generator import generate_answer

# -----------------------------
# Full RAG Pipeline
# -----------------------------

def query_rag(query, vectordb, all_docs=None, docs_pkl_path="docs.pkl"):
    """
    Complete RAG pipeline (Fast & Lightweight)
    """

    print("\n==============================")
    print("[>>] STARTING RAG PIPELINE")
    print("==============================")

    # -----------------------------
    # Step 1: Hybrid Search
    # -----------------------------
    # Skips multi-query generation for extreme speed
    hybrid_docs = hybrid_retrieval(query, vectordb, all_docs=all_docs, docs_pkl_path=docs_pkl_path)

    # -----------------------------
    # Step 2: Final Result Selection
    # -----------------------------
    # Skips CrossEncoder neural net re-ranking to save RAM & CPU
    final_docs = hybrid_docs[:5]

    # -----------------------------
    # Step 3: Generate Answer
    # -----------------------------
    answer = generate_answer(query, final_docs)

    print("\n[OK] RAG PIPELINE COMPLETED")

    return answer, final_docs