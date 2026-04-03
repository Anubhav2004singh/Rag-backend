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
You are an expert AI assistant that explains complex concepts beautifully, accurately, and in very simple, easy-to-understand language.
You are answering questions based EXCLUSIVELY on the provided context below.

CRITICAL INSTRUCTIONS:
1. NEVER cite or mention "Document X", "Source", "Context", or "Text". Explain the topic naturally as if you already know it.
2. Structure your response professionally. Provide a short introductory sentence, break the core points into bullet points, and offer a short logical conclusion.
3. Use rich Markdown formatting including Headers (##), Bold text, and clear spacing.
4. Keep your language simple and accessible, avoiding dense phrasing.
5. Provide a full, comprehensive, and highly specific answer. Extract as much detail as possible from the facts in the context.
6. If the answer is not present in the context, simply state: "The information you are asking for is not present in the uploaded document."

Context:
{context}

Question:
{query}

COMPREHENSIVE STRUCTURED FINAL ANSWER:
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    print("[OK] Answer generated")

    return response.content