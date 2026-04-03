from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)


# -----------------------------
# Load LLM (Gemini)
# -----------------------------

def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


# -----------------------------
# Build Context from Docs
# -----------------------------

def build_context(docs):
    """
    Combine retrieved documents into context
    """

    context_parts = []

    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        context_parts.append(f"[Document {i}]\n{content}")

    context = "\n\n".join(context_parts)

    return context


# -----------------------------
# Generate Final Answer
# -----------------------------

def generate_answer(query, docs):
    """
    Generate answer using retrieved documents
    """

    print("\n[AI] Generating final answer...")

    if not docs:
        return "No relevant documents found."

    llm = get_llm()

    context = build_context(docs)

    prompt = f"""
You are an AI assistant answering questions based ONLY on the provided context.

Instructions:
- Answer strictly from the context
- Do not make up information
- If answer is not present, say "Not found in document"
- Be clear and structured

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    print("[OK] Answer generated")

    return response.content