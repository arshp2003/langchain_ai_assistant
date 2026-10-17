"""
Tool approval policies for Human-in-the-Loop (HITL) workflow.

Defines which tools require human approval based on risk level.
Low-risk tools auto-approve, high-risk tools always require approval.
"""

from typing import Any


# Tool policy definitions
# priority: "low" | "medium" | "high"
# auto_approve: True means skip human approval for this tool
TOOL_POLICIES: dict[str, dict[str, Any]] = {
    # Low risk - read-only or safe operations
    "search_documents": {
        "priority": "low",
        "auto_approve": True,
        "description": "Search through uploaded documents",
    },
    "list_documents": {
        "priority": "low",
        "auto_approve": True,
        "description": "List all uploaded documents",
    },
    "list_memories": {
        "priority": "low",
        "auto_approve": True,
        "description": "List user memories",
    },
    "add_numbers": {
        "priority": "low",
        "auto_approve": True,
        "description": "Add two numbers",
    },
    "multiply_numbers": {
        "priority": "low",
        "auto_approve": True,
        "description": "Multiply two numbers",
    },
    "divide_numbers": {
        "priority": "low",
        "auto_approve": True,
        "description": "Divide two numbers",
    },
    
    # Medium risk - operations that could have moderate impact
    "extract_memory": {
        "priority": "medium",
        "auto_approve": True,  # Auto-approve unless extracting sensitive data
        "description": "Extract and save user memories",
        "conditions": {
            "max_memories_per_session": 10,  # Alert if extracting too many
        },
    },
    
    # High risk - destructive or irreversible operations
    "delete_document": {
        "priority": "high",
        "auto_approve": False,
        "description": "Delete an uploaded document and its vector embeddings permanently",
    },
    "delete_memory": {
        "priority": "high",
        "auto_approve": False,
        "description": "Delete a memory from long-term storage permanently",
    },
    "delete_file": {
        "priority": "high",
        "auto_approve": False,
        "description": "Delete a file permanently",
    },
    "send_email": {
        "priority": "high",
        "auto_approve": False,
        "description": "Send an email to someone",
    },
    "execute_code": {
        "priority": "high",
        "auto_approve": False,
        "description": "Execute arbitrary code",
    },
    "database_write": {
        "priority": "high",
        "auto_approve": False,
        "description": "Write to database",
    },
    "api_call_external": {
        "priority": "high",
        "auto_approve": False,
        "description": "Call external API",
    },
}


def get_tool_policy(tool_name: str) -> dict[str, Any]:
    """Get the policy for a specific tool."""
    return TOOL_POLICIES.get(tool_name, {
        "priority": "high",
        "auto_approve": False,
        "description": f"Unknown tool: {tool_name}",
    })


def should_require_approval(tool_name: str, tool_args: dict[str, Any] = None) -> bool:
    """
    Determine if a tool call requires human approval.
    
    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments being passed to the tool (for conditional checks)
    
    Returns:
        True if human approval is required, False if auto-approve
    """
    policy = get_tool_policy(tool_name)
    
    # High priority tools always require approval
    if policy["priority"] == "high":
        return not policy["auto_approve"]
    
    # Medium priority - check conditions
    if policy["priority"] == "medium" and "conditions" in policy:
        conditions = policy["conditions"]
        
        # Example: Check if extracting too many memories at once
        if "max_memories_per_session" in conditions:
            # This would need to be passed in tool_args or tracked in state
            estimated_count = tool_args.get("estimated_count", 0) if tool_args else 0
            if estimated_count > conditions["max_memories_per_session"]:
                return True
    
    # Low priority - auto-approve
    return False


def get_pending_tool_calls_requiring_approval(tool_calls: list[dict]) -> list[dict]:
    """
    Filter tool calls to only those requiring human approval.
    
    Args:
        tool_calls: List of tool call dicts with 'name' and 'args' keys
    
    Returns:
        List of tool calls that require approval
    """
    requires_approval = []
    auto_approved = []
    
    for tc in tool_calls:
        tool_name = tc.get("name", "unknown")
        tool_args = tc.get("args", {})
        
        if should_require_approval(tool_name, tool_args):
            requires_approval.append(tc)
        else:
            auto_approved.append(tc)
    
    return requires_approval


def format_approval_request(tool_calls: list[dict]) -> str:
    """
    Format a human-readable approval request for pending tool calls.
    
    Args:
        tool_calls: List of tool calls requiring approval
    
    Returns:
        Formatted string describing the pending actions
    """
    if not tool_calls:
        return "No actions require approval."
    
    lines = ["The following actions require your approval:"]
    
    for i, tc in enumerate(tool_calls, 1):
        tool_name = tc.get("name", "unknown")
        tool_args = tc.get("args", {})
        policy = get_tool_policy(tool_name)
        
        lines.append(f"\n{i}. **{tool_name}** ({policy['priority']} risk)")
        lines.append(f"   Description: {policy['description']}")
        lines.append(f"   Arguments: {tool_args}")
    
    return "\n".join(lines)
