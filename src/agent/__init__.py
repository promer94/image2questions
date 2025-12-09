"""
LangChain Agent Orchestration for Question Extraction.

This module provides a LangChain-based agent with short-term memory for
extracting questions from images and saving them to various formats.

The agent uses:
- LangChain's `create_agent` for orchestration
- `InMemorySaver` from LangGraph for short-term memory (conversation history)
- Custom tools for image analysis, JSON/Word generation, and validation

Classes:
    QuestionExtractionAgent: Main agent class with conversation memory

Functions:
    create_question_extraction_agent: Factory function for creating agents
    extract_questions: Quick single-turn extraction
    get_system_prompt: Get system prompt in specified language

Example:
    >>> from src.agent import QuestionExtractionAgent
    >>> 
    >>> # Create agent with short-term memory
    >>> agent = QuestionExtractionAgent()
    >>> 
    >>> # Multi-turn conversation (agent remembers context)
    >>> response = agent.chat("提取 test.jpg 中的选择题")
    >>> print(response)
    >>> response = agent.chat("验证这些题目")  # References previous extraction
    >>> print(response)
    >>> response = agent.chat("保存为Word文档")  # Saves the validated questions
    >>> print(response)
    >>> 
    >>> # Start a new conversation (clear memory)
    >>> agent.new_conversation()
"""

from .agent import (
    QuestionExtractionAgent,
    create_question_extraction_agent,
    extract_questions,
)
from .prompts import (
    get_system_prompt,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_ZH,
    SYSTEM_PROMPT_EN,
)
from .middleware import (
    BatchProcessingContextMiddleware,
)

__all__ = [
    # Agent class
    "QuestionExtractionAgent",
    # Factory functions
    "create_question_extraction_agent",
    "extract_questions",
    # Prompts
    "get_system_prompt",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_ZH",
    "SYSTEM_PROMPT_EN",
    # Middleware
    "BatchProcessingContextMiddleware",
]
