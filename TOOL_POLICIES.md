# Tool Approval Policy System

## Overview

The HITL (Human-in-the-Loop) system now uses **risk-based tool approval** instead of requiring approval for all tools. Low-risk tools auto-approve, high-risk tools require human approval.

## How It Works

### 1. Tool Policy Definitions (`backend/tool_policies.py`)

Each tool has a policy with:
- **priority**: `"low"`, `"medium"`, or `"high"`
- **auto_approve**: `True` = skip human approval, `False` = require approval
- **description**: Human-readable description
- **conditions** (optional): Dynamic checks (e.g., max rows affected)

```python
TOOL_POLICIES = {
    "search_documents": {"priority": "low", "auto_approve": True},
    "add_numbers": {"priority": "low", "auto_approve": True},
    "extract_memory": {"priority": "medium", "auto_approve": True},
    "delete_file": {"priority": "high", "auto_approve": False},
    "send_email": {"priority": "high", "auto_approve": False},
}
```

### 2. Approval Flow

1. **Agent decides to call tools** → Routes to `check_tool_approval` node
2. **Check tool policies** → Determines which tools need approval
3. **If all tools are low-risk** → Auto-approve and execute
4. **If any tool is high-risk** → Interrupt and show approval panel

### 3. Graph Structure

```
START → agent → check_tool_approval → tools → agent → validate_answer → extract_memory → END
                                      ↓
                              (if needs approval)
                                      ↓
                              [INTERRUPT for human review]
```

## API Response Examples

### Auto-Approved (Low-Risk Tools)

```json
{
  "answer": "8",
  "thread_id": "run-abc123",
  "pending_approval": {
    "status": "auto_approved",
    "thread_id": "run-abc123"
  }
}
```

### Requires Approval (High-Risk Tools)

```json
{
  "answer": "",
  "thread_id": "run-abc123",
  "pending_approval": {
    "status": "awaiting_approval",
    "pending_node": "tools",
    "tool_calls": [
      {"name": "delete_file", "args": {"filename": "important.pdf"}}
    ],
    "auto_approved_tools": [
      {"name": "search_documents", "args": {"query": "find file"}}
    ]
  }
}
```

## Adding New Tools

1. Define the tool in `backend/agent_graph.py`:
```python
@tool
def my_new_tool(param: str) -> str:
    """Tool description."""
    return result
```

2. Add it to the `tools` list:
```python
tools = [..., my_new_tool]
```

3. Add policy in `backend/tool_policies.py`:
```python
TOOL_POLICIES = {
    ...
    "my_new_tool": {
        "priority": "medium",
        "auto_approve": True,
        "description": "Does something moderately risky",
    },
}
```

## Customizing Policies

### Make a tool always require approval:
```python
"my_tool": {"priority": "high", "auto_approve": False}
```

### Add conditional approval:
```python
"database_write": {
    "priority": "medium",
    "auto_approve": True,
    "conditions": {
        "max_rows_affected": 100,
    },
}
```

### Auto-approve everything (disable HITL for tools):
```python
# Set all tools to auto_approve: True
```

## Real-World Best Practices

1. **Start restrictive** - Default to `auto_approve: False` for new tools
2. **Monitor usage** - Check logs to see which tools are frequently approved
3. **Gradually auto-approve** - Move tools to auto-approve after verifying safety
4. **Use conditions** - For medium-risk tools, add thresholds (e.g., max amount, max rows)
5. **Audit trail** - Keep logs of all approvals/rejections for compliance

## Testing

Test with different tool types:

```bash
# Low-risk (auto-approve)
curl -X POST http://127.0.0.1:8000/chat/hitl \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 5 + 3?", "user_id": "user-123"}'

# High-risk (requires approval) - if you add a delete_file tool
curl -X POST http://127.0.0.1:8000/chat/hitl \
  -H "Content-Type: application/json" \
  -d '{"question": "Delete the file test.txt", "user_id": "user-123"}'
```

## Frontend Behavior

- **Auto-approved**: Shows "✓ Tools were auto-approved (low-risk)" message
- **Requires approval**: Shows approval panel with Approve/Reject buttons
- **Mixed**: Shows only high-risk tools in approval panel, mentions auto-approved tools
