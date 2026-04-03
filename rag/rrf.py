# -----------------------------
# RRF (Reciprocal Rank Fusion)
# -----------------------------

def rrf_fusion(result_lists, k=60):
    """
    Combine multiple ranked document lists using RRF

    Args:
        result_lists: list of lists of documents
        k: constant (default 60)

    Returns:
        ranked_docs (list)
    """

    print(flush=True, "\n[LINK] Running RRF Fusion...")

    scores = {}
    doc_map = {}

    # Loop over each result list
    for docs in result_lists:
        for rank, doc in enumerate(docs):

            doc_id = hash(doc.page_content)

            # Store doc reference
            doc_map[doc_id] = doc

            # Initialize score
            if doc_id not in scores:
                scores[doc_id] = 0

            # RRF scoring formula
            scores[doc_id] += 1 / (k + rank + 1)

    # Sort by score (descending)
    ranked_doc_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Convert back to documents
    ranked_docs = [doc_map[doc_id] for doc_id, _ in ranked_doc_ids]

    print(flush=True, f"[OK] RRF combined docs: {len(ranked_docs)}")

    return ranked_docs