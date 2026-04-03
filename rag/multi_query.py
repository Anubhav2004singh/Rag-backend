from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)

import os
print("API KEY:", os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# Gemini LLM
# -----------------------------

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


# -----------------------------
# Generate Multiple Queries
# -----------------------------

def generate_multi_queries(query):

    llm = get_llm()

    prompt = f"""
Generate 5 different versions of the following user question.

Return ONLY plain text queries, one per line.
DO NOT return JSON, markdown, or numbering.

Question:
{query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    raw = response.content

    # Clean output
    lines = raw.split("\n")

    queries = []
    for line in lines:
        line = line.strip()

        # remove unwanted formatting
        if not line:
            continue
        if line.startswith("```") or line.startswith("[") or line.startswith("]"):
            continue

        # remove numbering like "1. "
        if "." in line:
            parts = line.split(".", 1)
            if parts[0].isdigit():
                line = parts[1].strip()

        queries.append(line)

    return queries


# -----------------------------
# Retrieve Documents from DB
# -----------------------------

def retrieve_documents(queries, vectordb, k=5):
    """
    Retrieve documents for each query
    """

    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    all_docs = []

    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    return all_docs


# -----------------------------
# Remove Duplicate Docs
# -----------------------------

def deduplicate_docs(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        content = doc.page_content

        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    return unique_docs


# -----------------------------
# MAIN FUNCTION (Plug & Play)
# -----------------------------

def multi_query_retrieval(query, vectordb):
    """
    Full custom multi-query pipeline
    """

    print("\n[MULTI] Generating multiple queries...")

    queries = generate_multi_queries(query)

    print("Generated Queries:")
    for q in queries:
        print(" -", q)

    # include original query also
    queries.append(query)

    print("\n[SEARCH] Retrieving documents...")

    docs = retrieve_documents(queries, vectordb)

    print(f"Retrieved {len(docs)} docs before deduplication")

    docs = deduplicate_docs(docs)

    print(f"[OK] Final docs after deduplication: {len(docs)}")

    return docs