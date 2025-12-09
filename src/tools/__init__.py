"""
LangChain Tools for Question Extraction Agent.

This module provides a collection of tools for extracting questions from images,
saving them to JSON and Word formats, and validating the results.

All tools are implemented using the LangChain @tool decorator and can be
directly used with LangChain agents.

Tools:
    - analyze_image: Extract questions from images and save to JSON file
    - load_questions_json: Load questions from JSON file
    - validate_questions_tool: Validate question quality
    - batch_process_images: Process multiple images from a directory

Example:
    from langchain.agents import create_agent
    from src.tools import get_all_tools
    
    tools = get_all_tools()
    agent = create_agent(model="gpt-4o", tools=tools)
"""

from .image_analysis import analyze_image
from .validation import validate_questions_tool
from .batch_processor import batch_process_images

# Also export result models for programmatic use
from .base import (
    ToolResult,
    ImageAnalysisResult,
    FileOperationResult,
    ValidationResult,
    BatchProcessingResult,
)


def get_all_tools() -> list:
    """Get all available tools for the agent.
    
    Returns:
        List of LangChain tool functions ready to use with an agent.
    
    Example:
        from langchain.agents import create_agent
        from src.tools import get_all_tools
        
        tools = get_all_tools()
        agent = create_agent(
            model="gpt-4o",
            tools=tools,
            system_prompt="You are a helpful assistant..."
        )
    """
    return [
        analyze_image,
        validate_questions_tool,
        batch_process_images,
    ]


def get_extraction_tools() -> list:
    """Get only image extraction tools.
    
    Returns:
        List of tools for extracting questions from images.
    """
    return [
        analyze_image,
        batch_process_images,
    ]


def get_output_tools() -> list:
    """Get only file output tools.
    
    Returns:
        List of tools for loading questions from files.
    """
    return [
    ]


def get_validation_tools() -> list:
    """Get only validation tools.
    
    Returns:
        List of tools for validating questions.
    """
    return [
        validate_questions_tool,
    ]


__all__ = [
    # Main tools
    "analyze_image",
    "validate_questions_tool",
    "batch_process_images",
    # Tool getters
    "get_all_tools",
    "get_extraction_tools",
    "get_output_tools",
    "get_validation_tools",
    # Result models
    "ToolResult",
    "ImageAnalysisResult",
    "FileOperationResult",
    "ValidationResult",
    "BatchProcessingResult",
]
