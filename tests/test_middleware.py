"""
Tests for custom middleware in the agent module.

This module tests the ContextCleanupMiddleware and BatchProcessingContextMiddleware
to ensure they correctly remove obsolete tool messages from the conversation context.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.messages import AIMessage, ToolMessage, HumanMessage

from src.agent.middleware import ContextCleanupMiddleware, BatchProcessingContextMiddleware


class TestContextCleanupMiddleware:
    """Tests for ContextCleanupMiddleware."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.middleware = ContextCleanupMiddleware()
    
    def test_cleanup_rules_defined(self):
        """Test that cleanup rules are properly defined."""
        assert "save_questions_json" in self.middleware.CLEANUP_RULES
        assert "batch_process_images" in self.middleware.CLEANUP_RULES
        assert "analyze_image" in self.middleware.CLEANUP_RULES["save_questions_json"]
        assert "save_questions_json" in self.middleware.CLEANUP_RULES["batch_process_images"]
    
    def test_no_cleanup_for_untracked_tools(self):
        """Test that untracked tools don't trigger cleanup."""
        # Create a mock request for a tool not in CLEANUP_RULES
        mock_request = MagicMock()
        mock_request.tool_call = {"name": "some_other_tool", "id": "call_123"}
        mock_request.state = {"messages": []}
        
        # Create a mock handler that returns a ToolMessage
        mock_result = ToolMessage(content="result", tool_call_id="call_123")
        mock_handler = MagicMock(return_value=mock_result)
        
        # Execute
        result = self.middleware.wrap_tool_call(mock_request, mock_handler)
        
        # Verify handler was called and result returned as-is
        mock_handler.assert_called_once_with(mock_request)
        assert result == mock_result
    
    def test_cleanup_after_save_questions_json(self):
        """Test that analyze_image messages are cleaned up after save_questions_json."""
        # Create mock messages that simulate a conversation
        ai_msg_with_analyze = AIMessage(
            content="",
            id="ai_msg_1",
            tool_calls=[{"name": "analyze_image", "id": "call_analyze_1", "args": {}}]
        )
        tool_msg_analyze = ToolMessage(
            content="Analysis result",
            tool_call_id="call_analyze_1",
            id="tool_msg_1"
        )
        
        mock_request = MagicMock()
        mock_request.tool_call = {"name": "save_questions_json", "id": "call_save_1"}
        mock_request.state = {
            "messages": [
                HumanMessage(content="Process this image"),
                ai_msg_with_analyze,
                tool_msg_analyze,
            ]
        }
        
        # Create a mock handler
        mock_result = ToolMessage(content="Saved!", tool_call_id="call_save_1")
        mock_handler = MagicMock(return_value=mock_result)
        
        # Execute
        result = self.middleware.wrap_tool_call(mock_request, mock_handler)
        
        # Verify handler was called
        mock_handler.assert_called_once_with(mock_request)
        
        # Result should be a Command with message removals
        # Since messages have IDs, cleanup should have been triggered
        from langgraph.types import Command
        assert isinstance(result, Command)
        assert "messages" in result.update
    
    def test_cleanup_after_batch_process_images(self):
        """Test that save_questions_json messages are cleaned up after batch_process_images."""
        # Create mock messages
        ai_msg_with_save = AIMessage(
            content="",
            id="ai_msg_2",
            tool_calls=[{"name": "save_questions_json", "id": "call_save_1", "args": {}}]
        )
        tool_msg_save = ToolMessage(
            content="Saved!",
            tool_call_id="call_save_1",
            id="tool_msg_2"
        )
        
        mock_request = MagicMock()
        mock_request.tool_call = {"name": "batch_process_images", "id": "call_batch_1"}
        mock_request.state = {
            "messages": [
                HumanMessage(content="Process batch"),
                ai_msg_with_save,
                tool_msg_save,
            ]
        }
        
        # Create a mock handler
        mock_result = ToolMessage(content="Batch complete!", tool_call_id="call_batch_1")
        mock_handler = MagicMock(return_value=mock_result)
        
        # Execute
        result = self.middleware.wrap_tool_call(mock_request, mock_handler)
        
        # Verify cleanup was triggered
        from langgraph.types import Command
        assert isinstance(result, Command)


class TestBatchProcessingContextMiddleware:
    """Tests for BatchProcessingContextMiddleware."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.middleware = BatchProcessingContextMiddleware(keep_recent=2)
    
    def test_keep_recent_initialization(self):
        """Test that keep_recent is properly initialized."""
        assert self.middleware.keep_recent == 2
        
        # Test default value
        default_middleware = BatchProcessingContextMiddleware()
        assert default_middleware.keep_recent == 3
    
    def test_no_cleanup_when_under_limit(self):
        """Test that no cleanup happens when tool calls are under the limit."""
        # Create messages with only 2 tool calls (at limit)
        messages = [
            HumanMessage(content="Process"),
            AIMessage(
                content="",
                id="ai_1",
                tool_calls=[{"name": "analyze_image", "id": "call_1", "args": {}}]
            ),
            ToolMessage(content="Result 1", tool_call_id="call_1", id="tool_1"),
            AIMessage(
                content="",
                id="ai_2",
                tool_calls=[{"name": "analyze_image", "id": "call_2", "args": {}}]
            ),
            ToolMessage(content="Result 2", tool_call_id="call_2", id="tool_2"),
        ]
        
        state = {"messages": messages}
        mock_runtime = MagicMock()
        
        result = self.middleware.before_model(state, mock_runtime)
        
        # Should return None (no cleanup needed)
        assert result is None
    
    def test_cleanup_when_over_limit(self):
        """Test that cleanup happens when tool calls exceed the limit."""
        # Create messages with 4 analyze_image calls (over limit of 2)
        messages = [
            HumanMessage(content="Process"),
            AIMessage(
                content="",
                id="ai_1",
                tool_calls=[{"name": "analyze_image", "id": "call_1", "args": {}}]
            ),
            ToolMessage(content="Result 1", tool_call_id="call_1", id="tool_1"),
            AIMessage(
                content="",
                id="ai_2",
                tool_calls=[{"name": "analyze_image", "id": "call_2", "args": {}}]
            ),
            ToolMessage(content="Result 2", tool_call_id="call_2", id="tool_2"),
            AIMessage(
                content="",
                id="ai_3",
                tool_calls=[{"name": "analyze_image", "id": "call_3", "args": {}}]
            ),
            ToolMessage(content="Result 3", tool_call_id="call_3", id="tool_3"),
            AIMessage(
                content="",
                id="ai_4",
                tool_calls=[{"name": "analyze_image", "id": "call_4", "args": {}}]
            ),
            ToolMessage(content="Result 4", tool_call_id="call_4", id="tool_4"),
        ]
        
        state = {"messages": messages}
        mock_runtime = MagicMock()
        
        result = self.middleware.before_model(state, mock_runtime)
        
        # Should return a dict with messages to remove
        assert result is not None
        assert "messages" in result
        # Should have 4 RemoveMessage objects (2 AI messages + 2 tool messages)
        # for the oldest 2 calls
        from langchain.messages import RemoveMessage
        remove_messages = result["messages"]
        assert all(isinstance(msg, RemoveMessage) for msg in remove_messages)
        # Should remove messages for the first 2 tool calls (keeping the last 2)
        removed_ids = {msg.id for msg in remove_messages}
        assert "ai_1" in removed_ids or "tool_1" in removed_ids
        assert "ai_2" in removed_ids or "tool_2" in removed_ids
    
    def test_separate_cleanup_per_tool_type(self):
        """Test that cleanup is tracked separately for different tool types."""
        # Create messages with different tool types
        messages = [
            HumanMessage(content="Process"),
            # 3 analyze_image calls
            AIMessage(
                content="",
                id="ai_analyze_1",
                tool_calls=[{"name": "analyze_image", "id": "call_a1", "args": {}}]
            ),
            ToolMessage(content="Result", tool_call_id="call_a1", id="tool_a1"),
            AIMessage(
                content="",
                id="ai_analyze_2",
                tool_calls=[{"name": "analyze_image", "id": "call_a2", "args": {}}]
            ),
            ToolMessage(content="Result", tool_call_id="call_a2", id="tool_a2"),
            AIMessage(
                content="",
                id="ai_analyze_3",
                tool_calls=[{"name": "analyze_image", "id": "call_a3", "args": {}}]
            ),
            ToolMessage(content="Result", tool_call_id="call_a3", id="tool_a3"),
            # 2 save_questions_json calls (at limit)
            AIMessage(
                content="",
                id="ai_save_1",
                tool_calls=[{"name": "save_questions_json", "id": "call_s1", "args": {}}]
            ),
            ToolMessage(content="Saved", tool_call_id="call_s1", id="tool_s1"),
            AIMessage(
                content="",
                id="ai_save_2",
                tool_calls=[{"name": "save_questions_json", "id": "call_s2", "args": {}}]
            ),
            ToolMessage(content="Saved", tool_call_id="call_s2", id="tool_s2"),
        ]
        
        state = {"messages": messages}
        mock_runtime = MagicMock()
        
        result = self.middleware.before_model(state, mock_runtime)
        
        # Should only clean up analyze_image calls (3 > 2)
        # save_questions_json is at limit (2 == 2)
        assert result is not None
        from langchain.messages import RemoveMessage
        removed_ids = {msg.id for msg in result["messages"]}
        
        # The oldest analyze_image call should be removed
        assert "ai_analyze_1" in removed_ids or "tool_a1" in removed_ids
        
        # save_questions_json should NOT be removed
        assert "ai_save_1" not in removed_ids
        assert "tool_s1" not in removed_ids


class TestMiddlewareIntegration:
    """Integration tests for middleware."""
    
    def test_middleware_can_be_instantiated(self):
        """Test that middlewares can be instantiated without errors."""
        cleanup = ContextCleanupMiddleware()
        batch = BatchProcessingContextMiddleware()
        
        assert cleanup is not None
        assert batch is not None
    
    def test_middleware_has_required_methods(self):
        """Test that middlewares have the required methods."""
        cleanup = ContextCleanupMiddleware()
        batch = BatchProcessingContextMiddleware()
        
        # ContextCleanupMiddleware should have wrap_tool_call
        assert hasattr(cleanup, "wrap_tool_call")
        assert callable(cleanup.wrap_tool_call)
        
        # BatchProcessingContextMiddleware should have before_model
        assert hasattr(batch, "before_model")
        assert callable(batch.before_model)
