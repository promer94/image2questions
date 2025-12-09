"""
JSON Generator Tool for LangChain Agent.

This tool loads questions from JSON files.
The save functionality has been merged into analyze_image tool.
"""

import json
from pathlib import Path

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


# Export for convenient access
__all__ = [
    "load_existing_questions",
]
