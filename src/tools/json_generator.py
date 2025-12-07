"""
JSON Generator Tool for LangChain Agent.

This tool saves extracted questions to JSON files with support for
append mode and pretty formatting.
"""

import json
import os
from pathlib import Path
from typing import Any

from langchain.tools import tool


def parse_questions_input(questions_json: str) -> tuple[list[dict] | dict, str | None]:
    """Parse questions from JSON string input.
    
    Supports two formats:
    1. Array format: [{"title": "...", ...}, ...]
    2. Mixed format: {"multiple_choice": [...], "true_false": [...]}
    
    Args:
        questions_json: JSON string containing questions
        
    Returns:
        Tuple of (questions data, error message or None)
    """
    try:
        data = json.loads(questions_json)
        # Support both array format and mixed format from image analysis
        if isinstance(data, list):
            return data, None
        elif isinstance(data, dict):
            # Check for mixed format
            if "multiple_choice" in data or "true_false" in data:
                return data, None
            return [], "Invalid format: dictionary must contain 'multiple_choice' or 'true_false' keys"
        return [], "Questions must be a JSON array or mixed format object"
    except json.JSONDecodeError as e:
        return [], f"Invalid JSON: {str(e)}"


def load_existing_questions(file_path: Path) -> tuple[list[dict] | dict, str | None]:
    """Load existing questions from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (questions data, error message or None)
    """
    if not file_path.exists():
        return [], None
    
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, list):
            return data, None
        elif isinstance(data, dict):
            # Support mixed format
            if "multiple_choice" in data or "true_false" in data:
                return data, None
            return [], "Existing file has invalid format"
        return [], "Existing file does not contain a JSON array or valid mixed format"
    except json.JSONDecodeError as e:
        return [], f"Existing file has invalid JSON: {str(e)}"
    except Exception as e:
        return [], f"Error reading file: {str(e)}"


def save_questions_to_json(
    questions: list[dict] | dict,
    output_path: Path,
    pretty: bool = True
) -> tuple[bool, str]:
    """Save questions to a JSON file.
    
    Args:
        questions: List of question dictionaries or mixed format dict
        output_path: Path to save the JSON file
        pretty: Whether to format with indentation
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format JSON
        if pretty:
            content = json.dumps(questions, ensure_ascii=False, indent=2)
        else:
            content = json.dumps(questions, ensure_ascii=False)
        
        # Write file
        output_path.write_text(content, encoding="utf-8")
        
        # Count questions for message
        if isinstance(questions, dict):
            count = len(questions.get("multiple_choice", [])) + len(questions.get("true_false", []))
        else:
            count = len(questions)
        
        return True, f"Saved {count} questions to {output_path}"
        
    except Exception as e:
        return False, f"Error saving file: {str(e)}"


@tool
def save_questions_json(
    questions_json: str,
    output_path: str,
    append: bool = False,
    pretty: bool = True,
    question_type: str = "auto"
) -> str:
    """Save extracted questions to a JSON file.
    
    This tool saves question data to a JSON file. It supports both
    overwrite and append modes for building up question banks.
    
    Args:
        questions_json: JSON string containing questions. Supports two formats:
                       1. Array format: [{"title": "...", "options": {...}}] or [{"title": "..."}]
                       2. Mixed format from image analysis: {"multiple_choice": [...], "true_false": [...]}
        output_path: File path where the JSON will be saved.
                    Will create parent directories if needed.
        append: If True, append questions to existing file.
               If False, overwrite existing file. Default: False
        pretty: If True, format JSON with indentation. Default: True
        question_type: Which questions to extract from mixed format:
                      - "auto": Save all questions (for mixed, save as mixed format)
                      - "multiple_choice": Extract only multiple choice questions
                      - "true_false": Extract only true/false questions
    
    Returns:
        A string describing the result of the operation.
    """
    # Parse input questions
    data, error = parse_questions_input(questions_json)
    if error:
        return f"Error: {error}"
    
    # Handle mixed format
    if isinstance(data, dict):
        mc_questions = data.get("multiple_choice", [])
        tf_questions = data.get("true_false", [])
        
        if question_type == "multiple_choice":
            questions = mc_questions
        elif question_type == "true_false":
            questions = tf_questions
        else:  # auto - save as mixed format or flatten
            # Save in mixed format to preserve structure
            if not mc_questions and not tf_questions:
                return "Error: No questions provided. Both multiple_choice and true_false arrays are empty."
            questions = data  # Keep the mixed format
    else:
        questions = data
    
    # Check if questions is empty
    if isinstance(questions, list) and not questions:
        return "Error: No questions provided. The questions array is empty."
    if isinstance(questions, dict) and not questions.get("multiple_choice") and not questions.get("true_false"):
        return "Error: No questions provided. Both arrays are empty."
    
    # Convert to Path object
    file_path = Path(output_path)
    
    # Ensure .json extension
    if file_path.suffix.lower() != ".json":
        file_path = file_path.with_suffix(".json")
    
    # Count questions for reporting
    if isinstance(questions, dict):
        q_count = len(questions.get("multiple_choice", [])) + len(questions.get("true_false", []))
    else:
        q_count = len(questions)
    
    # Handle append mode
    if append and file_path.exists():
        existing, error = load_existing_questions(file_path)
        if error:
            return f"Error loading existing file: {error}"
        
        # For mixed format appending to list, flatten to list
        if isinstance(questions, dict) and isinstance(existing, list):
            # Flatten mixed to list
            flat_questions = questions.get("multiple_choice", []) + questions.get("true_false", [])
            combined = existing + flat_questions
        elif isinstance(questions, list) and isinstance(existing, list):
            combined = existing + questions
        else:
            return "Error: Cannot append mixed format to non-list file or vice versa"
        
        success, message = save_questions_to_json(combined, file_path, pretty)
        
        if success:
            return f"Appended {q_count} questions to existing {len(existing)} questions.\n{message}"
        return message
    
    # Save (overwrite or new file)
    success, message = save_questions_to_json(questions, file_path, pretty)
    
    if success:
        return f"{message}"
    
    return message


@tool
def load_questions_json(file_path: str) -> str:
    """Load questions from a JSON file.
    
    This tool reads a JSON file containing questions and returns them
    for further processing or display.
    
    Args:
        file_path: Path to the JSON file to load.
    
    Returns:
        A string containing the questions in JSON format, or an error message.
    """
    path = Path(file_path)
    
    if not path.exists():
        return f"Error: File not found: {file_path}"
    
    if not path.is_file():
        return f"Error: Not a file: {file_path}"
    
    questions, error = load_existing_questions(path)
    if error:
        return f"Error: {error}"
    
    if not questions:
        return f"File is empty or contains no questions: {file_path}"
    
    result = f"Loaded {len(questions)} questions from {file_path}\n\n"
    result += json.dumps(questions, ensure_ascii=False, indent=2)
    
    return result


# Export for convenient access
__all__ = [
    "save_questions_json",
    "load_questions_json",
]
