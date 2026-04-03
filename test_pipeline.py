import time
import os
from rag.ingestion import run_complete_ingestion_pipeline
from rag.vectorstore import create_vector_store, load_vectorstore
from rag.query_pipeline import query_rag

def generate_pdf():
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello world. This is a special test document about the Antigravity Project, which is a super-secret initiative to cancel gravity.")
    doc.save("test.pdf")
    doc.close()

if __name__ == "__main__":
    generate_pdf()
    
    t0 = time.time()
    chunks = run_complete_ingestion_pipeline("test.pdf")
    assert len(chunks) > 0, "No chunks generated"
    create_vector_store(chunks, "test_collection")
    t1 = time.time()
    print(f"Ingestion time: {t1-t0:.2f}s")
    
    # query
    import pickle
    with open("db/docs_pkl/test_collection.pkl", "wb") as f:
        pickle.dump(chunks, f)
        
    t2 = time.time()
    db = load_vectorstore("test_collection")
    with open("db/docs_pkl/test_collection.pkl", "rb") as f:
        all_docs = pickle.load(f)
    print("Asking query...")
    ans, docs = query_rag("What is the Antigravity Project?", db, all_docs=all_docs)
    t3 = time.time()
    print("Answer:", ans)
    print(f"Query time: {t3-t2:.2f}s")
    print("Pipeline Test SUCCESS")
