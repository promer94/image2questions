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


def parse_questions_input(questions_json: str) -> tuple[dict, str | None]:
    """Parse questions from JSON string input.
    
    Only supports the unified dictionary format:
    {
        "type": "...",
        "multiple_choice": [...],
        "true_false": [...]
    }
    
    Args:
        questions_json: JSON string containing questions
        
    Returns:
        Tuple of (questions data, error message or None)
    """
    try:
        data = json.loads(questions_json)
        if isinstance(data, dict):
            # Check for valid keys
            if "multiple_choice" in data or "true_false" in data:
                return data, None
            return {}, "Invalid format: dictionary must contain 'multiple_choice' or 'true_false' keys"
        return {}, "Questions must be a JSON object (dictionary)"
    except json.JSONDecodeError as e:
        return {}, f"Invalid JSON: {str(e)}"


def load_existing_questions(file_path: Path) -> tuple[dict, str | None]:
    """Load existing questions from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (questions data, error message or None)
    """
    if not file_path.exists():
        return {}, None
    
    try:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, dict):
            if "multiple_choice" in data or "true_false" in data:
                return data, None
            return {}, "Existing file has invalid format"
        return {}, "Existing file does not contain a valid JSON object"
    except json.JSONDecodeError as e:
        return {}, f"Existing file has invalid JSON: {str(e)}"
    except Exception as e:
        return {}, f"Error reading file: {str(e)}"


def save_questions_to_json(
    questions: dict,
    output_path: Path,
    pretty: bool = True
) -> tuple[bool, str]:
    """Save questions to a JSON file.
    
    Args:
        questions: Question dictionary
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
        count = len(questions.get("multiple_choice", [])) + len(questions.get("true_false", []))
        
        return True, f"Saved {count} questions to {output_path}"
        
    except Exception as e:
        return False, f"Error saving file: {str(e)}"


@tool
def save_questions_json(
    questions_json: str,
    output_path: str,
    append: bool = False,
    pretty: bool = True,
    question_type: str = "auto",
    processed_images: list[str] | None = None
) -> str:
    """Save extracted questions to a JSON file.
    
    This tool saves question data to a JSON file. It supports both
    overwrite and append modes for building up question banks.
    
    Args:
        questions_json: JSON string containing questions in the unified dictionary format:
                       {"multiple_choice": [...], "true_false": [...]}
        output_path: File path where the JSON will be saved.
                    Will create parent directories if needed.
        append: If True, append questions to existing file.
               If False, overwrite existing file. Default: False
        pretty: If True, format JSON with indentation. Default: True
        question_type: Which questions to extract from mixed format:
                      - "auto": Save all questions (for mixed, save as mixed format)
                      - "multiple_choice": Extract only multiple choice questions
                      - "true_false": Extract only true/false questions
        processed_images: List of image paths that were processed to generate these questions.
                         These will be added to the 'processed_images' field in the output.
    
    Returns:
        A string describing the result of the operation.
    """
    # Parse input questions
    questions, error = parse_questions_input(questions_json)
    if error:
        return f"Error: {error}"
    
    # Handle filtering by type
    mc_questions = questions.get("multiple_choice", [])
    tf_questions = questions.get("true_false", [])
    
    if question_type == "multiple_choice":
        questions = {"multiple_choice": mc_questions, "true_false": []}
    elif question_type == "true_false":
        questions = {"multiple_choice": [], "true_false": tf_questions}
    
    # Check if questions is empty
    if not questions.get("multiple_choice") and not questions.get("true_false"):
        return "Error: No questions provided. Both arrays are empty."
    
    # Add processed images if provided
    if processed_images:
        questions["processed_images"] = processed_images

    # Convert to Path object
    file_path = Path(output_path)
    
    # Ensure .json extension
    if file_path.suffix.lower() != ".json":
        file_path = file_path.with_suffix(".json")
    
    # Count questions for reporting
    q_count = len(questions.get("multiple_choice", [])) + len(questions.get("true_false", []))
    
    # Handle append mode
    if append and file_path.exists():
        existing, error = load_existing_questions(file_path)
        if error:
            return f"Error loading existing file: {error}"
        
        # Merge existing and new questions
        combined = {
            "multiple_choice": existing.get("multiple_choice", []) + questions.get("multiple_choice", []),
            "true_false": existing.get("true_false", []) + questions.get("true_false", [])
        }
        
        # Merge processed images
        existing_images = existing.get("processed_images", [])
        new_images = questions.get("processed_images", [])
        if existing_images or new_images:
            # Use set to avoid duplicates, but keep order if possible? 
            # Sets don't keep order. Let's just append and unique-ify.
            all_images = existing_images + [img for img in new_images if img not in existing_images]
            combined["processed_images"] = all_images

        # Preserve type if present
        if "type" in existing:
            combined["type"] = existing["type"]
        elif "type" in questions:
            combined["type"] = questions["type"]
        
        success, message = save_questions_to_json(combined, file_path, pretty)
        
        if success:
            existing_count = len(existing.get("multiple_choice", [])) + len(existing.get("true_false", []))
            return f"Appended {q_count} questions to existing {existing_count} questions.\n{message}"
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
