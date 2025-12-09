"""
LangChain Agent with Short-Term Memory for Question Extraction.

This module provides a LangChain agent implementation using create_agent
with InMemorySaver for short-term memory (conversation history).

The agent can:
- Extract questions from images
- Save questions to JSON and Word formats
- Validate question quality
- Process images in batch
- Remember conversation context within a session
"""

import uuid
from typing import Optional, Union

from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from ..models.config import get_settings
from ..models.questions import AgentResponse
from ..tools import get_all_tools
from .prompts import get_system_prompt, SYSTEM_PROMPT



class QuestionExtractionAgent:
    """
    Question Extraction Agent with short-term memory.
    
    This agent uses LangChain's create_agent to orchestrate tools for
    extracting questions from images. It maintains conversation history
    within a session using InMemorySaver as a checkpointer.
    
    Attributes:
        agent: The compiled LangChain agent
        checkpointer: InMemorySaver for short-term memory
        thread_id: Current conversation thread ID
        
    Example:
        >>> agent = QuestionExtractionAgent()
        >>> response = agent.chat("从 test.jpg 中提取题目")
        >>> print(response)
        "我已成功从 test.jpg 中提取了 5 道选择题..."
        
        >>> # Agent remembers the previous context
        >>> response = agent.chat("保存为Word文档")
        >>> print(response)
        "已将刚才提取的 5 道题目保存到 output.docx"
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        language: str = "zh",
        use_structured_output: bool = True,
        checkpointer: Optional[object] = "default",
    ):
        """
        Initialize the Question Extraction Agent.
        
        Args:
            model_name: LLM model name (defaults to config)
            api_key: API key for the LLM (defaults to config)
            base_url: Base URL for OpenAI-compatible API (defaults to config)
            temperature: LLM temperature (defaults to config)
            system_prompt: Custom system prompt (defaults to built-in prompt)
            language: Language for default prompt ('zh' or 'en')
            use_structured_output: Whether to use structured output format (default True)
            checkpointer: Checkpointer for memory (defaults to InMemorySaver, set to None to disable)
        """
        settings = get_settings()
        
        # Configure LLM
        self.model = ChatOpenAI(
            model=model_name or settings.agent_model,
            api_key=api_key or settings.effective_agent_api_key,
            base_url=base_url or settings.agent_base_url,
            temperature=temperature if temperature is not None else settings.agent_temperature,
            ###use_responses_api=True,
            ### workaroud https://github.com/langchain-ai/langchain/issues/34124#issuecomment-3586763122
            ###output_version="responses/v1",
            ###use_previous_response_id=True
        )
        
        # Get tools
        self.tools = get_all_tools()
        
        # Configure system prompt
        self.system_prompt = system_prompt or get_system_prompt(language)
        
        # Create checkpointer for short-term memory
        # InMemorySaver stores conversation history in memory
        # This enables multi-turn conversations within a session
        if checkpointer == "default":
            self.checkpointer = InMemorySaver()
        else:
            self.checkpointer = checkpointer
        
        # Store structured output preference
        self.use_structured_output = use_structured_output

        # Configure middleware to manage context window
        # ClearToolUsesEdit removes old tool outputs to save tokens
        # ContextCleanupMiddleware removes specific tool messages after checkpoints:
        #   - Remove analyze_image messages after save_questions_json
        #   - Remove save_questions_json messages after batch_process_images
        middleware = [
           ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=8192,
                    keep=8,
                    clear_tool_inputs=True
                ),
            ],
          )
        ]
        
        # Create the agent using LangChain's create_agent
        # This builds a ReAct agent that runs on LangGraph
        # Use ToolStrategy for structured output to ensure stable responses
        agent_kwargs = {
            "model": self.model,
            "tools": self.tools,
            "system_prompt": self.system_prompt,
            "checkpointer": self.checkpointer,
            "middleware": middleware,
        }
        
        if use_structured_output:
            # ToolStrategy uses tool calling to generate structured output
            # This works with any model that supports tool calling (including Doubao)
            agent_kwargs["response_format"] = ToolStrategy(AgentResponse)
        
        self.agent = create_agent(**agent_kwargs)
        
        # Initialize thread ID for conversation tracking
        self._thread_id: Optional[str] = None
    
    @property
    def thread_id(self) -> str:
        """Get or create the current thread ID."""
        if self._thread_id is None:
            self._thread_id = str(uuid.uuid4())
        return self._thread_id
    
    def new_conversation(self) -> str:
        """
        Start a new conversation (clear memory context).
        
        Returns:
            New thread ID
        """
        self._thread_id = str(uuid.uuid4())
        return self._thread_id
    
    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
    ) -> Union[str, AgentResponse]:
        """
        Send a message to the agent and get a response.
        
        The agent maintains conversation history within the same thread_id,
        allowing it to reference previous messages and results.
        
        Args:
            message: User message
            thread_id: Optional thread ID for conversation (uses current if not provided)
            
        Returns:
            If use_structured_output is True: AgentResponse object
            Otherwise: Agent response as string
            
        Example:
            >>> agent = QuestionExtractionAgent()
            >>> print(agent.chat("提取 image.jpg 中的选择题"))
            "我从 image.jpg 中提取了 3 道选择题..."
            >>> print(agent.chat("验证这些题目的质量"))
            "这 3 道题目的验证结果如下..."
        """
        # Use provided thread_id or default to current conversation
        current_thread_id = thread_id or self.thread_id
        
        # Invoke the agent with message and thread configuration
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": current_thread_id}},
        )
        
        # If using structured output, return the structured response
        if self.use_structured_output and "structured_response" in result:
            structured_response = result["structured_response"]
            if structured_response is not None:
                return structured_response
        
        # Fallback: Extract the final response from the agent
        # The result contains a list of messages, we want the last AI message
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            # Handle different message types
            if hasattr(last_message, "content"):
                # If structured output was expected but not found, try to parse the content
                if self.use_structured_output:
                    try:
                        import json
                        content = last_message.content
                        if isinstance(content, str):
                            data = json.loads(content)
                            return AgentResponse(**data)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # If parsing fails, return a default structured response
                        return AgentResponse(
                            success=True,
                            message=last_message.content,
                            operation="chat"
                        )
                return last_message.content
            return str(last_message)
        
        # Return appropriate type based on structured output setting
        if self.use_structured_output:
            return AgentResponse(
                success=False,
                message="No response generated.",
                operation="unknown"
            )
        return "No response generated."
    
    def invoke(
        self,
        message: str,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Invoke the agent and return the full result.
        
        This is similar to chat() but returns the complete result dict
        including all messages, tool calls, and structured_response (if enabled).
        
        Args:
            message: User message
            thread_id: Optional thread ID for conversation
            
        Returns:
            Full result dictionary from the agent
        """
        current_thread_id = thread_id or self.thread_id
        
        return self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": current_thread_id}},
        )
    
    def stream(
        self,
        message: str,
        thread_id: Optional[str] = None,
    ):
        """
        Stream the agent response.
        
        Yields chunks of the agent's response as they are generated.
        
        Args:
            message: User message
            thread_id: Optional thread ID for conversation
            
        Yields:
            Response chunks from the agent
        """
        current_thread_id = thread_id or self.thread_id
        
        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": current_thread_id}},
        ):
            yield chunk
    
    def get_conversation_history(self, thread_id: Optional[str] = None) -> list:
        """
        Get the conversation history for a thread.
        
        Args:
            thread_id: Thread ID to get history for (uses current if not provided)
            
        Returns:
            List of messages in the conversation
        """
        current_thread_id = thread_id or self.thread_id
        
        # Get the state from the checkpointer
        try:
            state = self.agent.get_state(
                config={"configurable": {"thread_id": current_thread_id}}
            )
            if state and state.values:
                return state.values.get("messages", [])
        except Exception:
            pass
        
        return []


def create_question_extraction_agent(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
    language: str = "zh",
    use_structured_output: bool = True,
) -> QuestionExtractionAgent:
    """
    Factory function to create a Question Extraction Agent.
    
    This is a convenience function that creates and returns a configured
    QuestionExtractionAgent instance.
    
    Args:
        model_name: LLM model name (defaults to config)
        api_key: API key for the LLM (defaults to config)
        base_url: Base URL for OpenAI-compatible API (defaults to config)
        temperature: LLM temperature (defaults to config)
        system_prompt: Custom system prompt
        language: Language for default prompt ('zh' or 'en')
        use_structured_output: Whether to use structured output format (default True)
        
    Returns:
        Configured QuestionExtractionAgent instance
        
    Example:
        >>> agent = create_question_extraction_agent()
        >>> response = agent.chat("从 images/ 目录批量提取题目")
    """
    return QuestionExtractionAgent(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        system_prompt=system_prompt,
        language=language,
        use_structured_output=use_structured_output,
    )


# Convenience function for quick single-turn interactions
def extract_questions(
    message: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    use_structured_output: bool = True,
) -> Union[str, AgentResponse]:
    """
    Quick function for single-turn question extraction.
    
    This creates a temporary agent and processes a single request.
    For multi-turn conversations, use QuestionExtractionAgent instead.
    
    Args:
        message: User message describing the extraction task
        model_name: Optional LLM model name
        api_key: Optional API key
        use_structured_output: Whether to return structured output (default True)
        
    Returns:
        If use_structured_output is True: AgentResponse object
        Otherwise: Agent response string
        
    Example:
        >>> result = extract_questions("提取 test.jpg 中的题目并保存为 output.json")
    """
    agent = create_question_extraction_agent(
        model_name=model_name,
        api_key=api_key,
        use_structured_output=use_structured_output,
    )
    return agent.chat(message)
