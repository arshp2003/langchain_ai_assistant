from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

llm = ChatOpenRouter(
    model="qwen/qwen3.8-27b:free"
)

response = llm.invoke("Explain RAG in simple words.")

print(response.content)