import pickle
from rag.ingestion import run_complete_ingestion_pipeline
from rag.vectorstore import create_vector_store
import os
os.environ["HF_HUB_TIMEOUT"] = "60"
os.environ["HF_HUB_READ_TIMEOUT"] = "60"

pdf_path = r"D:\maal-masala\DL Project\Main Rag\rag-backend\docs\attention-is-all-you-need.pdf"

docs = run_complete_ingestion_pipeline(pdf_path)

# ✅ SAVE DOCS (VERY IMPORTANT)
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)

vectordb = create_vector_store(docs)


