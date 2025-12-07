"""
Image processing utilities for the question extraction agent.

This module provides functions for encoding, validating, and managing image files.
"""

import base64
from pathlib import Path
from typing import Iterator

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# MIME type mapping for supported image formats
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def encode_image_to_base64(image_path: str | Path) -> str:
    """
    Encode an image file to a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the file is not a valid image
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not is_valid_image(path):
        raise ValueError(f"Invalid image file: {image_path}")
    
    with open(path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str | Path) -> str:
    """
    Get the MIME type for an image based on its file extension.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type string (e.g., 'image/jpeg')
    """
    suffix = Path(image_path).suffix.lower()
    return MIME_TYPES.get(suffix, "image/jpeg")


def is_valid_image(image_path: str | Path) -> bool:
    """
    Check if a file is a valid image based on extension and existence.
    
    Args:
        image_path: Path to check
        
    Returns:
        True if the path points to a valid image file
    """
    path = Path(image_path)
    
    if not path.exists():
        return False
    
    if not path.is_file():
        return False
    
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def get_image_paths_from_directory(
    directory: str | Path,
    recursive: bool = False
) -> list[Path]:
    """
    Get all image file paths from a directory.
    
    Args:
        directory: Directory to search for images
        recursive: If True, search subdirectories recursively
        
    Returns:
        List of Path objects pointing to image files
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        NotADirectoryError: If the path is not a directory
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    def iter_images() -> Iterator[Path]:
        pattern = "**/*" if recursive else "*"
        for file_path in dir_path.glob(pattern):
            if is_valid_image(file_path):
                yield file_path
    
    # Sort by name for consistent ordering
    return sorted(iter_images(), key=lambda p: p.name.lower())


def build_image_content(image_paths: list[str | Path]) -> list[dict]:
    """
    Build the content array for a vision model API call with multiple images.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        List of content dictionaries for the API call
        
    Raises:
        ValueError: If no valid images are provided
    """
    content = []
    valid_count = 0
    
    for image_path in image_paths:
        try:
            base64_image = encode_image_to_base64(image_path)
            mime_type = get_image_mime_type(image_path)
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })
            valid_count += 1
        except (FileNotFoundError, ValueError) as e:
            # Log warning but continue with other images
            print(f"Warning: Skipping invalid image {image_path}: {e}")
    
    if valid_count == 0:
        raise ValueError("No valid images provided")
    
    return content


def get_image_info(image_path: str | Path) -> dict:
    """
    Get basic information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    path = Path(image_path)
    
    return {
        "path": str(path.absolute()),
        "name": path.name,
        "extension": path.suffix.lower(),
        "mime_type": get_image_mime_type(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "exists": path.exists(),
        "is_valid": is_valid_image(path),
    }
