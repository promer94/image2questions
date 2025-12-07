"""
Batch Processing Tool for LangChain Agent.

This tool processes multiple images from a directory,
extracting questions from all of them in a single operation.
"""

import json
import os
from pathlib import Path
from typing import Literal

from langchain.tools import tool

from .image_analysis import (
    extract_multiple_choice,
    extract_true_false,
    validate_image_paths,
)
from ..models.config import get_settings
from openai import OpenAI


# ==================== Supported Extensions ====================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def find_images_in_directory(
    directory: Path,
    recursive: bool = False
) -> list[str]:
    """Find all supported image files in a directory.
    
    Args:
        directory: Path to the directory
        recursive: If True, search subdirectories too
        
    Returns:
        List of image file paths
    """
    images = []
    
    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            images.extend(str(p) for p in directory.rglob(f"*{ext}"))
            images.extend(str(p) for p in directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            images.extend(str(p) for p in directory.glob(f"*{ext}"))
            images.extend(str(p) for p in directory.glob(f"*{ext.upper()}"))
    
    # Sort for consistent ordering
    return sorted(set(images))


def process_images_batch(
    image_paths: list[str],
    question_type: str,
    client: OpenAI,
    model: str,
    batch_size: int = 5
) -> tuple[list[dict], list[str]]:
    """Process images in batches.
    
    Args:
        image_paths: List of image paths to process
        question_type: "multiple_choice" or "true_false"
        client: OpenAI client instance
        model: Model name to use
        batch_size: Number of images per batch
        
    Returns:
        Tuple of (all_questions, errors)
    """
    all_questions = []
    errors = []
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        
        try:
            if question_type == "multiple_choice":
                questions = extract_multiple_choice(client, model, batch)
            else:
                questions = extract_true_false(client, model, batch)
            
            all_questions.extend(questions)
            
        except Exception as e:
            error_msg = f"Error processing batch {i//batch_size + 1}: {str(e)}"
            errors.append(error_msg)
            
            # Try processing images individually if batch fails
            for path in batch:
                try:
                    if question_type == "multiple_choice":
                        questions = extract_multiple_choice(client, model, [path])
                    else:
                        questions = extract_true_false(client, model, [path])
                    all_questions.extend(questions)
                except Exception as e2:
                    errors.append(f"Error processing {path}: {str(e2)}")
    
    return all_questions, errors


# ==================== LangChain Tool ====================

@tool
def batch_process_images(
    directory_path: str,
    question_type: str = "multiple_choice",
    recursive: bool = False,
    batch_size: int = 5
) -> str:
    """Process all images in a directory to extract questions.
    
    This tool scans a directory for image files and extracts questions
    from all of them. It processes images in batches for efficiency
    and provides detailed progress information.
    
    Args:
        directory_path: Path to the directory containing images.
                       Supported formats: jpg, jpeg, png, gif, webp, bmp
        question_type: Type of questions to extract. Must be one of:
                      - "multiple_choice": Questions with options A, B, C, D
                      - "true_false": Statement questions for true/false judgment
        recursive: If True, also search in subdirectories.
                  Default: False
        batch_size: Number of images to process in each batch.
                   Larger batches are faster but use more memory.
                   Default: 5
    
    Returns:
        A string containing:
        - Total images found and processed
        - Total questions extracted
        - Any errors encountered
        - The extracted questions in JSON format
    """
    # Validate directory
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        return f"Error: Directory not found: {directory_path}"
    
    if not dir_path.is_dir():
        return f"Error: Not a directory: {directory_path}"
    
    # Validate question type
    if question_type not in ("multiple_choice", "true_false"):
        return f"Error: Invalid question_type '{question_type}'. Must be 'multiple_choice' or 'true_false'."
    
    # Find images
    image_paths = find_images_in_directory(dir_path, recursive)
    
    if not image_paths:
        return f"No images found in {directory_path}. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
    
    # Validate batch size
    batch_size = max(1, min(batch_size, 10))  # Clamp between 1 and 10
    
    try:
        # Get settings and create client
        settings = get_settings()
        client = OpenAI(
            api_key=settings.doubao_api_key,
            base_url=settings.doubao_base_url,
        )
        
        # Process images
        all_questions, errors = process_images_batch(
            image_paths=image_paths,
            question_type=question_type,
            client=client,
            model=settings.doubao_model,
            batch_size=batch_size
        )
        
        # Build result message
        lines = [
            f"Batch Processing Complete",
            "=" * 60,
            f"Directory: {directory_path}",
            f"Recursive: {'Yes' if recursive else 'No'}",
            f"Question Type: {question_type.replace('_', ' ')}",
            "",
            f"Results:",
            f"  - Images found: {len(image_paths)}",
            f"  - Questions extracted: {len(all_questions)}",
            f"  - Errors: {len(errors)}",
        ]
        
        if errors:
            lines.append("")
            lines.append("Errors encountered:")
            for error in errors[:10]:  # Limit error display
                lines.append(f"  - {error}")
            if len(errors) > 10:
                lines.append(f"  ... and {len(errors) - 10} more errors")
        
        lines.append("")
        lines.append("Extracted questions:")
        lines.append(json.dumps(all_questions, ensure_ascii=False, indent=2))
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error during batch processing: {str(e)}"


@tool
def list_images_in_directory(
    directory_path: str,
    recursive: bool = False
) -> str:
    """List all image files in a directory.
    
    This tool scans a directory for image files and returns a list
    of found images. Useful for previewing what will be processed
    before running batch extraction.
    
    Args:
        directory_path: Path to the directory to scan.
        recursive: If True, also search in subdirectories.
                  Default: False
    
    Returns:
        A string containing the list of found images.
    """
    # Validate directory
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        return f"Error: Directory not found: {directory_path}"
    
    if not dir_path.is_dir():
        return f"Error: Not a directory: {directory_path}"
    
    # Find images
    image_paths = find_images_in_directory(dir_path, recursive)
    
    if not image_paths:
        return f"No images found in {directory_path}. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
    
    # Build result
    lines = [
        f"Found {len(image_paths)} image(s) in {directory_path}",
        f"Recursive: {'Yes' if recursive else 'No'}",
        "",
        "Images:"
    ]
    
    for path in image_paths:
        # Show relative path if possible
        try:
            rel_path = Path(path).relative_to(dir_path)
            lines.append(f"  - {rel_path}")
        except ValueError:
            lines.append(f"  - {path}")
    
    return "\n".join(lines)


# Export for convenient access
__all__ = [
    "batch_process_images",
    "list_images_in_directory",
    "find_images_in_directory",
]
