from sentence_transformers import CrossEncoder

# -----------------------------
# Lazy-loaded Reranker Model
# -----------------------------

_reranker_model = None

def _get_reranker():
    """Load reranker model on first use (lazy loading)"""
    global _reranker_model
    if _reranker_model is None:
        print(flush=True, "[LOAD] Loading reranker model...")
        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print(flush=True, "[OK] Reranker loaded")
    return _reranker_model


# -----------------------------
# Rerank Documents
# -----------------------------

def rerank_documents(query, docs, top_k=5):
    """
    Rerank documents using CrossEncoder

    Args:
        query: user query
        docs: list of documents
        top_k: number of final docs

    Returns:
        reranked_docs (top_k)
    """

    print(flush=True, "\n[RANK] Running Reranker...")

    if not docs:
        print(flush=True, "[WARN] No documents to rerank")
        return []

    reranker_model = _get_reranker()

    # Create (query, doc) pairs
    pairs = [(query, doc.page_content) for doc in docs]

    # Predict relevance scores
    scores = reranker_model.predict(pairs)

    # Combine docs + scores
    scored_docs = list(zip(docs, scores))

    # Sort by score (descending)
    ranked_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    # Extract top_k docs
    top_docs = [doc for doc, _ in ranked_docs[:top_k]]

    print(flush=True, f"[OK] Reranked top {top_k} documents")

    return top_docs