"""
File system utilities for the question extraction agent.

This module provides functions for file and directory operations,
including JSON reading/writing and path management.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def ensure_directory(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_json_file(file_path: str | Path) -> Any:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(
    file_path: str | Path,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    append: bool = False
) -> Path:
    """
    Write data to a JSON file.
    
    Args:
        file_path: Path to the output file
        data: Data to write (must be JSON-serializable)
        indent: Number of spaces for indentation
        ensure_ascii: If True, escape non-ASCII characters
        append: If True and file exists with a list, append to it
        
    Returns:
        Path to the written file
    """
    path = Path(file_path)
    
    # Ensure parent directory exists
    ensure_directory(path.parent)
    
    # Handle append mode for list data
    if append and path.exists() and isinstance(data, list):
        try:
            existing_data = read_json_file(path)
            if isinstance(existing_data, list):
                data = existing_data + data
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is invalid or doesn't exist, just write new data
            pass
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
    
    return path


def get_unique_filename(
    directory: str | Path,
    base_name: str,
    extension: str
) -> Path:
    """
    Generate a unique filename by appending a number if necessary.
    
    Args:
        directory: Directory for the file
        base_name: Base name for the file (without extension)
        extension: File extension (with or without leading dot)
        
    Returns:
        Path to a unique filename
        
    Example:
        >>> get_unique_filename("./output", "questions", ".json")
        # Returns: ./output/questions.json (if doesn't exist)
        # Returns: ./output/questions_1.json (if questions.json exists)
    """
    dir_path = Path(directory)
    ensure_directory(dir_path)
    
    # Normalize extension
    if not extension.startswith("."):
        extension = f".{extension}"
    
    # Try base name first
    candidate = dir_path / f"{base_name}{extension}"
    if not candidate.exists():
        return candidate
    
    # Add numbers until we find a unique name
    counter = 1
    while True:
        candidate = dir_path / f"{base_name}_{counter}{extension}"
        if not candidate.exists():
            return candidate
        counter += 1


def is_valid_path(path: str | Path) -> bool:
    """
    Check if a path string is valid (not necessarily existing).
    
    Args:
        path: Path to validate
        
    Returns:
        True if the path is valid
    """
    try:
        Path(path).resolve()
        return True
    except (OSError, ValueError):
        return False


def get_file_info(file_path: str | Path) -> dict:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    path = Path(file_path)
    
    info = {
        "path": str(path.absolute()),
        "name": path.name,
        "stem": path.stem,
        "extension": path.suffix,
        "exists": path.exists(),
        "is_file": path.is_file() if path.exists() else False,
        "is_directory": path.is_dir() if path.exists() else False,
    }
    
    if path.exists() and path.is_file():
        stat = path.stat()
        info.update({
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        })
    
    return info


def backup_file(file_path: str | Path, backup_dir: Optional[str | Path] = None) -> Optional[Path]:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        backup_dir: Directory for backups (defaults to same directory)
        
    Returns:
        Path to the backup file, or None if original doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        return None
    
    if backup_dir:
        backup_directory = ensure_directory(backup_dir)
    else:
        backup_directory = path.parent
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
    backup_path = backup_directory / backup_name
    
    # Copy the file
    backup_path.write_bytes(path.read_bytes())
    
    return backup_path
