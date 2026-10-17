from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openrouter import ChatOpenRouter

from embeddings import embeddings


# -----------------------------
# 1. Load Chroma
# -----------------------------

vector_store = Chroma(
    collection_name="langchain_notes",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# -----------------------------
# 3. Retriever
# -----------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------
# 4. Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI assistant that answers questions
using the provided context.

Rules:
- Use the context to answer.
- Do not make up information.
- If the answer is not in the context, say:
  "I don't know based on the provided documents."

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])


# -----------------------------
# 5. Model
# -----------------------------

model = ChatOpenRouter(
    model="qwen/qwen3.8-27b:free"
)


# -----------------------------
# 6. Parser
# -----------------------------

parser = StrOutputParser()


# -----------------------------
# 7. Format documents
# -----------------------------

def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )


# -----------------------------
# 8. LCEL RAG Chain
# -----------------------------

rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | parser
)


# -----------------------------
# 9. Ask question
# -----------------------------

question = "What is LangChain?"

answer = rag_chain.invoke(question)

print("\n========== ANSWER ==========\n")
print(answer)