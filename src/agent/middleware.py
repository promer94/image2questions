"""
Custom Middleware for Context Management.

This module provides custom middleware to manage the conversation context
by removing obsolete tool-use messages after specific tool calls complete.

The middleware helps reduce context window usage by:
1. Removing `analyze_image` tool messages after `save_questions_json` completes
2. Removing `save_questions_json` tool messages after `batch_process_images` completes

This is particularly useful during batch processing where multiple images
are analyzed sequentially, and we want to keep the context focused on
the current batch rather than accumulating all historical tool calls.
"""


from typing import Any, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import RemoveMessage, AIMessage, ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command



class ContextCleanupMiddleware(AgentMiddleware):
    """
    Middleware that cleans up context after specific tool calls.
    
    This middleware monitors tool executions and removes obsolete messages
    from the conversation history to keep the context window manageable.
    
    Cleanup Rules:
        - After `save_questions_json`: Remove all previous `analyze_image` tool messages
        - After `batch_process_images`: Remove all previous `save_questions_json` tool messages
    
    This follows a "checkpoint" pattern where completing a save operation
    allows us to discard the intermediate analysis results that are now
    persisted to disk.
    
    Example:
        from langchain.agents import create_agent
        from src.agent.middleware import ContextCleanupMiddleware
        
        agent = create_agent(
            model="gpt-4o",
            tools=tools,
            middleware=[ContextCleanupMiddleware()],
        )
    """
    
    # Mapping of trigger tool -> tools to clean up
    CLEANUP_RULES = {
        "save_questions_json": ["analyze_image"],
        "batch_process_images": ["save_questions_json"],
    }
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """
        Wrap tool calls to perform context cleanup after specific tools complete.
        
        Args:
            request: The tool call request containing tool name and arguments
            handler: The handler function to execute the tool
            
        Returns:
            The tool result (ToolMessage or Command)
        """
        # Get the tool name that will be executed
        tool_name = request.tool_call.get("name", "")
        tool_call_id = request.tool_call.get("id", "")
        
      
        
        # Execute the tool first
        result = handler(request)
        


        # Check if this tool triggers any cleanup
        if tool_name in self.CLEANUP_RULES:
            tools_to_clean = self.CLEANUP_RULES[tool_name]
          
            # Get current state to find messages to remove
            state = request.state
            messages = state.get("messages", [])
            
         
            
            # Find message IDs to remove
            # We need to remove:
            # 1. AIMessage with tool_calls for the specified tools
            # 2. ToolMessage responses for those tool calls
            messages_to_remove = []
            tool_call_ids_to_remove = set()
            
            for msg in messages:
                # Check if this is an AIMessage with tool calls
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        if tool_call.get("name") in tools_to_clean:
                            # Mark this message and its tool call ID for removal
                            if msg.id:
                                messages_to_remove.append(msg.id)
                               
                            tool_call_ids_to_remove.add(tool_call.get("id"))
                
                # Check if this is a ToolMessage that should be removed
                if isinstance(msg, ToolMessage):
                    # Remove if it's a response to a tool call we're cleaning up
                    if msg.tool_call_id in tool_call_ids_to_remove:
                        if msg.id:
                            messages_to_remove.append(msg.id)
                          
                    # Or if the tool name matches
                    elif hasattr(msg, "name") and msg.name in tools_to_clean:
                        if msg.id:
                            messages_to_remove.append(msg.id)
                           
            
            # If we have messages to remove, return a Command to update state
            if messages_to_remove:
                
                # Create RemoveMessage objects for each message to remove
                remove_messages = [RemoveMessage(id=msg_id) for msg_id in messages_to_remove]
                
                # Return a Command that includes both the result and the state update
                return Command(
                    update={
                        "messages": [result] + remove_messages
                    }
                )
        
        # No cleanup needed, return the result as-is
        return result

    def _find_messages_to_remove(
        self,
        tool_name: str,
        messages: list,
    ) -> list[str]:
        """
        Find message IDs that should be removed based on cleanup rules.
        
        Args:
            tool_name: The tool that was just executed
            messages: Current message list from state
            
        Returns:
            List of message IDs to remove
        """
        if tool_name not in self.CLEANUP_RULES:
            return []
        
        tools_to_clean = self.CLEANUP_RULES[tool_name]
        messages_to_remove = []
        tool_call_ids_to_remove = set()
        
        for msg in messages:
            # Check if this is an AIMessage with tool calls
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") in tools_to_clean:
                        if msg.id:
                            messages_to_remove.append(msg.id)
                        tool_call_ids_to_remove.add(tool_call.get("id"))
            
            # Check if this is a ToolMessage that should be removed
            if isinstance(msg, ToolMessage):
                if msg.tool_call_id in tool_call_ids_to_remove:
                    if msg.id:
                        messages_to_remove.append(msg.id)
                elif hasattr(msg, "name") and msg.name in tools_to_clean:
                    if msg.id:
                        messages_to_remove.append(msg.id)
        
        return messages_to_remove

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """
        Async version: Wrap tool calls to perform context cleanup after specific tools complete.
        
        Args:
            request: The tool call request containing tool name and arguments
            handler: The async handler function to execute the tool
            
        Returns:
            The tool result (ToolMessage or Command)
        """
        # Get the tool name that will be executed
        tool_name = request.tool_call.get("name", "")
        
        # Execute the tool first (await the async handler)
        result = await handler(request)
        
        # Check if this tool triggers any cleanup
        if tool_name in self.CLEANUP_RULES:
            state = request.state
            messages = state.get("messages", [])
            
            # Find messages to remove
            messages_to_remove = self._find_messages_to_remove(tool_name, messages)
            
            # If we have messages to remove, return a Command to update state
            if messages_to_remove:
                remove_messages = [RemoveMessage(id=msg_id) for msg_id in messages_to_remove]
                return Command(
                    update={
                        "messages": [result] + remove_messages
                    }
                )
        
        # No cleanup needed, return the result as-is
        return result


class BatchProcessingContextMiddleware(AgentMiddleware):
    """
    Alternative middleware implementation using before_model hook.
    
    This middleware cleans up old tool messages before each model call,
    keeping only the most recent tool results to maintain context relevance.
    
    Unlike ContextCleanupMiddleware which triggers on specific tool completions,
    this middleware runs before every model call and applies cleanup based on
    what tools have been recently executed.
    
    Example:
        from langchain.agents import create_agent
        from src.agent.middleware import BatchProcessingContextMiddleware
        
        agent = create_agent(
            model="gpt-4o",
            tools=tools,
            middleware=[BatchProcessingContextMiddleware(keep_recent=2)],
        )
    """
    
    def __init__(self, keep_recent: int = 3):
        """
        Initialize the middleware.
        
        Args:
            keep_recent: Number of recent tool call pairs to keep for each tool type
        """
        super().__init__()
        self.keep_recent = keep_recent
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Clean up old tool messages before model calls.
        
        This method examines the message history and removes old tool call/response
        pairs while keeping the most recent ones for context.
        
        Args:
            state: Current agent state containing messages
            runtime: Runtime context
            
        Returns:
            State update with RemoveMessage objects, or None if no cleanup needed
        """
        messages = state.get("messages", [])
        
      
        
        # Track tool calls by type
        tool_calls_by_type: dict[str, list[tuple[str, str]]] = {}  # tool_name -> [(ai_msg_id, tool_msg_id)]
        
        # Build a map of tool call IDs to their tool names
        tool_call_id_to_name: dict[str, str] = {}
        
        # First pass: identify all tool calls
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_call_id = tool_call.get("id", "")
                    if tool_name and tool_call_id:
                        tool_call_id_to_name[tool_call_id] = tool_name
                        if tool_name not in tool_calls_by_type:
                            tool_calls_by_type[tool_name] = []
                        tool_calls_by_type[tool_name].append((msg.id, tool_call_id))
        
        # Second pass: match tool messages to their calls
        tool_msg_ids: dict[str, str] = {}  # tool_call_id -> tool_msg_id
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                tool_msg_ids[msg.tool_call_id] = msg.id
        
        # Determine which messages to remove
        messages_to_remove = []
        
        for tool_name, calls in tool_calls_by_type.items():
            # Only keep the most recent N calls for each tool type
            if len(calls) > self.keep_recent:
                calls_to_remove = calls[:-self.keep_recent]
                
                for ai_msg_id, tool_call_id in calls_to_remove:
                    if ai_msg_id:
                        messages_to_remove.append(ai_msg_id)
                    if tool_call_id in tool_msg_ids and tool_msg_ids[tool_call_id]:
                        messages_to_remove.append(tool_msg_ids[tool_call_id])
        
        # Return state update if we have messages to remove
        if messages_to_remove:
            # Deduplicate and create RemoveMessage objects
            unique_ids = list(set(messages_to_remove))
           
            return {
                "messages": [RemoveMessage(id=msg_id) for msg_id in unique_ids]
            }
    
        return None

    def _compute_messages_to_remove(self, messages: list) -> list[str]:
        """
        Compute which message IDs should be removed based on keep_recent limit.
        
        Args:
            messages: Current message list from state
            
        Returns:
            List of unique message IDs to remove
        """
        # Track tool calls by type
        tool_calls_by_type: dict[str, list[tuple[str, str]]] = {}
        tool_call_id_to_name: dict[str, str] = {}
        
        # First pass: identify all tool calls
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_call_id = tool_call.get("id", "")
                    if tool_name and tool_call_id:
                        tool_call_id_to_name[tool_call_id] = tool_name
                        if tool_name not in tool_calls_by_type:
                            tool_calls_by_type[tool_name] = []
                        tool_calls_by_type[tool_name].append((msg.id, tool_call_id))
        
        # Second pass: match tool messages to their calls
        tool_msg_ids: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                tool_msg_ids[msg.tool_call_id] = msg.id
        
        # Determine which messages to remove
        messages_to_remove = []
        
        for tool_name, calls in tool_calls_by_type.items():
            if len(calls) > self.keep_recent:
                calls_to_remove = calls[:-self.keep_recent]
                for ai_msg_id, tool_call_id in calls_to_remove:
                    if ai_msg_id:
                        messages_to_remove.append(ai_msg_id)
                    if tool_call_id in tool_msg_ids and tool_msg_ids[tool_call_id]:
                        messages_to_remove.append(tool_msg_ids[tool_call_id])
        
        return list(set(messages_to_remove))

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Async version: Clean up old tool messages before model calls.
        
        Args:
            state: Current agent state containing messages
            runtime: Runtime context
            
        Returns:
            State update with RemoveMessage objects, or None if no cleanup needed
        """
        messages = state.get("messages", [])
        unique_ids = self._compute_messages_to_remove(messages)
        
        if unique_ids:
            return {
                "messages": [RemoveMessage(id=msg_id) for msg_id in unique_ids]
            }
        
        return None


# Export for convenient access
__all__ = [
    "ContextCleanupMiddleware",
    "BatchProcessingContextMiddleware",
]
