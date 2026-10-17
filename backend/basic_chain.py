from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter


load_dotenv()


# 1. Chat Model
model = ChatOpenRouter(
    model="qwen/qwen3.8-27b:free"
)


# 2. Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI teacher who explains technical concepts simply."
    ),
    (
        "human",
        "Explain {topic} to a {level} student."
    )
])


# 3. Output Parser
parser = StrOutputParser()


# 4. LCEL Chain
chain = prompt | model | parser


# 5. Invoke the chain
response = chain.invoke({
    "topic": "RAG",
    "level": "beginner"
})


print(response)