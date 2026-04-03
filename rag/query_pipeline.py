from rag.multi_query import multi_query_retrieval
from rag.Hybrid_search import hybrid_retrieval
from rag.rrf import rrf_fusion
from rag.reranker import rerank_documents
from rag.generator import generate_answer


# -----------------------------
# Full RAG Pipeline
# -----------------------------

def query_rag(query, vectordb, all_docs=None, docs_pkl_path="docs.pkl"):
    """
    Complete RAG pipeline

    Args:
        query: user question
        vectordb: loaded vector database
        all_docs: pre-loaded documents for BM25 hybrid search (optional)
        docs_pkl_path: path to docs pickle file (used if all_docs is None)

    Returns:
        final answer string
    """

    print("\n==============================")
    print("[>>] STARTING RAG PIPELINE")
    print("==============================")

    # -----------------------------
    # Step 1: Multi Query
    # -----------------------------
    multi_docs = multi_query_retrieval(query, vectordb)

    # -----------------------------
    # Step 2: Hybrid Search
    # -----------------------------
    hybrid_docs = hybrid_retrieval(query, vectordb, all_docs=all_docs, docs_pkl_path=docs_pkl_path)

    # -----------------------------
    # Step 3: RRF Fusion
    # -----------------------------
    fused_docs = rrf_fusion([multi_docs, hybrid_docs])

    # -----------------------------
    # Step 4: Reranker
    # -----------------------------
    reranked_docs = rerank_documents(query, fused_docs, top_k=5)

    # -----------------------------
    # Step 5: Generate Answer
    # -----------------------------
    answer = generate_answer(query, reranked_docs)

    print("\n[OK] RAG PIPELINE COMPLETED")

    return answer, reranked_docs