from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# -----------------------------
# 1. Define tools
# -----------------------------

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


tools = [
    add_numbers,
    multiply_numbers,
]


# -----------------------------
# 2. Create model
# -----------------------------

model = ChatOpenRouter(
    model="qwen/qwen3.8-27b:free"
)

model_with_tools = model.bind_tools(tools)


# -----------------------------
# 3. Define state
# -----------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# 4. Define nodes
# -----------------------------

def call_model(state: AgentState) -> dict:
    """Node that asks the model what to do next."""
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)


# -----------------------------
# 5. Build graph
# -----------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

agent = builder.compile()


# -----------------------------
# 6. Ask agent
# -----------------------------

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is 25 multiplied by 10?"
        }
    ]
})


# -----------------------------
# 7. Print final response
# -----------------------------

print("\n========== FINAL ANSWER ==========\n")

print(result["messages"][-1].content)