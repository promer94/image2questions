"""
Unit tests for the LangChain Agent module.

Tests cover:
- Agent initialization
- Prompt configuration
- Memory/checkpointer setup
- Tool integration
- Conversation flow
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import UUID

from src.agent import (
    QuestionExtractionAgent,
    create_question_extraction_agent,
    extract_questions,
    get_system_prompt,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_ZH,
    SYSTEM_PROMPT_EN,
)
from src.agent.prompts import get_system_prompt as prompts_get_system_prompt


# =============================================================================
# Test: Prompts
# =============================================================================

class TestPrompts:
    """Tests for system prompts."""
    
    def test_system_prompt_zh_exists(self):
        """Test that Chinese system prompt is defined."""
        assert SYSTEM_PROMPT_ZH is not None
        assert len(SYSTEM_PROMPT_ZH) > 100
        assert "试题" in SYSTEM_PROMPT_ZH or "题目" in SYSTEM_PROMPT_ZH
    
    def test_system_prompt_en_exists(self):
        """Test that English system prompt is defined."""
        assert SYSTEM_PROMPT_EN is not None
        assert len(SYSTEM_PROMPT_EN) > 100
        assert "question" in SYSTEM_PROMPT_EN.lower()
    
    def test_default_system_prompt_is_chinese(self):
        """Test that default prompt is Chinese."""
        assert SYSTEM_PROMPT == SYSTEM_PROMPT_ZH
    
    def test_get_system_prompt_zh(self):
        """Test getting Chinese prompt."""
        prompt = get_system_prompt("zh")
        assert prompt == SYSTEM_PROMPT_ZH
    
    def test_get_system_prompt_en(self):
        """Test getting English prompt."""
        prompt = get_system_prompt("en")
        assert prompt == SYSTEM_PROMPT_EN
    
    def test_get_system_prompt_default(self):
        """Test default language is Chinese."""
        prompt = get_system_prompt()
        assert prompt == SYSTEM_PROMPT_ZH
    
    def test_get_system_prompt_case_insensitive(self):
        """Test language code is case insensitive."""
        assert get_system_prompt("EN") == SYSTEM_PROMPT_EN
        assert get_system_prompt("ZH") == SYSTEM_PROMPT_ZH
    
    def test_prompts_contain_tool_descriptions(self):
        """Test that prompts mention all tools."""
        tools = [
            "analyze_image",
            "save_questions_json",
            "save_questions_word",
            "validate_questions",
            "batch_process",
            "list_images",
        ]
        # Check that tool concepts are mentioned (not exact names)
        for prompt in [SYSTEM_PROMPT_ZH, SYSTEM_PROMPT_EN]:
            assert "image" in prompt.lower() or "图片" in prompt
            assert "json" in prompt.lower() or "JSON" in prompt
            assert "word" in prompt.lower() or "Word" in prompt


# =============================================================================
# Test: Agent Initialization
# =============================================================================

class TestAgentInitialization:
    """Tests for QuestionExtractionAgent initialization."""
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_agent_init_with_defaults(self, mock_create_agent, mock_chat_openai):
        """Test agent initialization with default settings."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent()
        
        # Verify ChatOpenAI was called
        mock_chat_openai.assert_called_once()
        
        # Verify create_agent was called with tools
        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args[1]
        assert "tools" in call_kwargs
        assert "system_prompt" in call_kwargs
        assert "checkpointer" in call_kwargs
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_agent_init_with_custom_params(self, mock_create_agent, mock_chat_openai):
        """Test agent initialization with custom parameters."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent(
            model_name="custom-model",
            api_key="test-key",
            base_url="https://custom.api",
            temperature=0.5,
            system_prompt="Custom prompt",
        )
        
        # Verify ChatOpenAI was called with custom params
        mock_chat_openai.assert_called_once()
        call_kwargs = mock_chat_openai.call_args[1]
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://custom.api"
        assert call_kwargs["temperature"] == 0.5
        
        # Verify custom system prompt
        create_call_kwargs = mock_create_agent.call_args[1]
        assert create_call_kwargs["system_prompt"] == "Custom prompt"
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_agent_init_with_english_prompt(self, mock_create_agent, mock_chat_openai):
        """Test agent initialization with English prompt."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent(language="en")
        
        create_call_kwargs = mock_create_agent.call_args[1]
        assert create_call_kwargs["system_prompt"] == SYSTEM_PROMPT_EN
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_agent_has_checkpointer(self, mock_create_agent, mock_chat_openai):
        """Test that agent has InMemorySaver checkpointer."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent()
        
        # Verify checkpointer exists
        assert agent.checkpointer is not None
        
        # Verify checkpointer was passed to create_agent
        create_call_kwargs = mock_create_agent.call_args[1]
        assert create_call_kwargs["checkpointer"] is not None


# =============================================================================
# Test: Thread Management
# =============================================================================

class TestThreadManagement:
    """Tests for conversation thread management."""
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_thread_id_auto_generated(self, mock_create_agent, mock_chat_openai):
        """Test that thread_id is auto-generated on first access."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent()
        
        thread_id = agent.thread_id
        assert thread_id is not None
        # Should be a valid UUID string
        UUID(thread_id)  # Will raise if invalid
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_thread_id_consistent(self, mock_create_agent, mock_chat_openai):
        """Test that thread_id remains consistent within session."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent()
        
        thread_id1 = agent.thread_id
        thread_id2 = agent.thread_id
        
        assert thread_id1 == thread_id2
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_new_conversation_changes_thread_id(self, mock_create_agent, mock_chat_openai):
        """Test that new_conversation generates new thread_id."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent()
        
        thread_id1 = agent.thread_id
        new_thread_id = agent.new_conversation()
        thread_id2 = agent.thread_id
        
        assert new_thread_id == thread_id2
        assert thread_id1 != thread_id2


# =============================================================================
# Test: Chat Method
# =============================================================================

class TestChatMethod:
    """Tests for the chat method."""
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_chat_invokes_agent(self, mock_create_agent, mock_chat_openai):
        """Test that chat method invokes the agent."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Test response")]
        }
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        response = agent.chat("Test message")
        
        # Verify agent was invoked
        mock_agent.invoke.assert_called_once()
        
        # Check invoke arguments
        call_args = mock_agent.invoke.call_args
        assert call_args[0][0]["messages"][0]["role"] == "user"
        assert call_args[0][0]["messages"][0]["content"] == "Test message"
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_chat_uses_thread_id(self, mock_create_agent, mock_chat_openai):
        """Test that chat passes thread_id in config."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Response")]
        }
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        agent.chat("Test")
        
        # Check config contains thread_id
        call_kwargs = mock_agent.invoke.call_args[1]
        assert "config" in call_kwargs
        assert "configurable" in call_kwargs["config"]
        assert "thread_id" in call_kwargs["config"]["configurable"]
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_chat_extracts_response_content(self, mock_create_agent, mock_chat_openai):
        """Test that chat extracts content from response."""
        mock_message = MagicMock()
        mock_message.content = "Extracted response"
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_message]}
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        response = agent.chat("Test")
        
        assert response.message == "Extracted response"
        assert response.success == True
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_chat_handles_empty_response(self, mock_create_agent, mock_chat_openai):
        """Test chat handles empty response gracefully."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": []}
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        response = agent.chat("Test")
        
        assert response.message == "No response generated."
        assert response.success == False
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_chat_with_custom_thread_id(self, mock_create_agent, mock_chat_openai):
        """Test chat with custom thread_id."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Response")]
        }
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        agent.chat("Test", thread_id="custom-thread")
        
        call_kwargs = mock_agent.invoke.call_args[1]
        assert call_kwargs["config"]["configurable"]["thread_id"] == "custom-thread"


# =============================================================================
# Test: Invoke Method
# =============================================================================

class TestInvokeMethod:
    """Tests for the invoke method."""
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_invoke_returns_full_result(self, mock_create_agent, mock_chat_openai):
        """Test that invoke returns the full result dict."""
        expected_result = {
            "messages": [MagicMock(content="Response")],
            "some_other_key": "value"
        }
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = expected_result
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        result = agent.invoke("Test")
        
        assert result == expected_result


# =============================================================================
# Test: Stream Method
# =============================================================================

class TestStreamMethod:
    """Tests for the stream method."""
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_stream_yields_chunks(self, mock_create_agent, mock_chat_openai):
        """Test that stream method yields chunks."""
        chunks = ["chunk1", "chunk2", "chunk3"]
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter(chunks)
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        
        result_chunks = list(agent.stream("Test"))
        
        assert result_chunks == chunks


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_create_question_extraction_agent(self, mock_create_agent, mock_chat_openai):
        """Test create_question_extraction_agent factory."""
        mock_create_agent.return_value = MagicMock()
        
        agent = create_question_extraction_agent(
            model_name="test-model",
            api_key="test-key"
        )
        
        assert isinstance(agent, QuestionExtractionAgent)
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_extract_questions_convenience(self, mock_create_agent, mock_chat_openai):
        """Test extract_questions convenience function."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Extracted 5 questions")]
        }
        mock_create_agent.return_value = mock_agent
        
        result = extract_questions("Extract from test.jpg")
        
        assert result.message == "Extracted 5 questions"
        assert result.success == True


# =============================================================================
# Test: Tools Integration
# =============================================================================

class TestToolsIntegration:
    """Tests for tools integration with the agent."""
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_agent_has_all_tools(self, mock_create_agent, mock_chat_openai):
        """Test that agent receives all tools."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent()
        
        # Verify tools were passed to create_agent
        create_call_kwargs = mock_create_agent.call_args[1]
        tools = create_call_kwargs["tools"]
        
        # Should have 6 tools
        assert len(tools) == 6
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_tools_have_invoke_method(self, mock_create_agent, mock_chat_openai):
        """Test that all tools have invoke method (LangChain tools)."""
        mock_create_agent.return_value = MagicMock()
        
        agent = QuestionExtractionAgent()
        
        for tool in agent.tools:
            # LangChain tools are StructuredTool objects with invoke method
            assert hasattr(tool, "invoke") or hasattr(tool, "func")


# =============================================================================
# Test: Memory Behavior
# =============================================================================

class TestMemoryBehavior:
    """Tests for short-term memory behavior."""
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_same_thread_preserves_context(self, mock_create_agent, mock_chat_openai):
        """Test that same thread_id preserves conversation context."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Response")]
        }
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        
        # Multiple calls with same agent should use same thread
        agent.chat("Message 1")
        agent.chat("Message 2")
        
        # Verify both calls used the same thread_id
        calls = mock_agent.invoke.call_args_list
        thread_id_1 = calls[0][1]["config"]["configurable"]["thread_id"]
        thread_id_2 = calls[1][1]["config"]["configurable"]["thread_id"]
        
        assert thread_id_1 == thread_id_2
    
    @patch("src.agent.agent.ChatOpenAI")
    @patch("src.agent.agent.create_agent")
    def test_new_conversation_uses_new_thread(self, mock_create_agent, mock_chat_openai):
        """Test that new_conversation creates new thread."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Response")]
        }
        mock_create_agent.return_value = mock_agent
        
        agent = QuestionExtractionAgent()
        
        agent.chat("Message 1")
        agent.new_conversation()
        agent.chat("Message 2")
        
        # Verify different thread_ids
        calls = mock_agent.invoke.call_args_list
        thread_id_1 = calls[0][1]["config"]["configurable"]["thread_id"]
        thread_id_2 = calls[1][1]["config"]["configurable"]["thread_id"]
        
        assert thread_id_1 != thread_id_2
