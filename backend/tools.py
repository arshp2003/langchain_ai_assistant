from dotenv import load_dotenv

load_dotenv()

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool


# -----------------------------
# 1. Define the tool
# -----------------------------

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# -----------------------------
# 2. Create model
# -----------------------------

model = ChatOpenRouter(
    model="qwen/qwen3.8-27b:free"
)


# -----------------------------
# 3. Give tool to the model
# -----------------------------

model_with_tools = model.bind_tools(
    [add_numbers]
)


# -----------------------------
# 4. Ask the model
# -----------------------------

question = "What is 25 + 17?"

response = model_with_tools.invoke(question)

print("\n========== MODEL RESPONSE ==========\n")
print(response)

print("\n========== TOOL CALLS ==========\n")
print(response.tool_calls)

from langchain_core.messages import ToolMessage


if response.tool_calls:

    tool_call = response.tool_calls[0]

    # Execute the requested tool
    tool_result = add_numbers.invoke(
        tool_call["args"]
    )

    print("\n========== TOOL RESULT ==========\n")
    print(tool_result)

    # Send the original conversation + tool result back to the model
    final_response = model_with_tools.invoke([
        ("human", question),
        response,
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    ])

    print("\n========== FINAL ANSWER ==========\n")
    print(final_response.content)

else:

    print("\n========== FINAL ANSWER ==========\n")
    print(response.content)