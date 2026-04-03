from rag.vectorstore import load_vectorstore
from rag.query_pipeline import query_rag

# Load DB
vectordb = load_vectorstore()

query = "Explain attention mechanism in transformers"

answer = query_rag(query, vectordb)

print(flush=True, "\n[TIP] FINAL ANSWER:\n")
print(flush=True, answer)