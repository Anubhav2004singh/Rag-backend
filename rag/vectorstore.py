
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


DB_PATH = "db/chroma_db"


# -----------------------------
# Embedding Model
# -----------------------------

def get_embedding():

    return  GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# -----------------------------
# Create & Store Vector DB
# -----------------------------

def create_vector_store(documents, collection_name="default"):
    """
    Create vector store for documents

    Args:
        documents: list of LangChain Document objects
        collection_name: unique name for the collection (e.g. document ID)
    """

    print(flush=True, f"Creating embeddings and storing in ChromaDB (collection: {collection_name})...")

    embeddings = get_embedding()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(flush=True, "[OK] Vector store saved at:", DB_PATH)

    return vectorstore


# -----------------------------
# Load Existing Vector DB
# -----------------------------

def load_vectorstore(collection_name="default"):
    """
    Load existing vector store

    Args:
        collection_name: name of the collection to load
    """

    print(flush=True, f"Loading existing vector store (collection: {collection_name})...")

    embeddings = get_embedding()

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    print(flush=True, "[OK] Vector store loaded")

    return vectorstore
