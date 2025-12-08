"""Batch processing helper that discovers images and reports status.

This module provides tools to:
- discover image paths in a directory
- report on the current state of processing (images found vs questions saved)

The actual processing is delegated to the agent, which should use:
- analyze_image: to extract questions from images
- save_questions_json: to save the results
"""

from pathlib import Path
from typing import Literal

from langchain.tools import tool
from .json_generator import load_existing_questions
from .base import BatchProcessingResult


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


# ==================== LangChain Tool ====================

@tool
def batch_process_images(
    directory_path: str,
    recursive: bool = False,
    output_path: str | None = None,
    batch_size: int = 3,
) -> str:
    """Scan a directory for images and report processing status.
    
    This tool does NOT process the images automatically. Instead, it:
    1. Finds all images in the directory
    2. Checks the output JSON file (if it exists) for existing questions and processed images
    3. Returns a status report so the agent can decide which images to process
    
    Args:
        directory_path: Path to the directory containing images.
        recursive: If True, also search in subdirectories.
        output_path: Path to the JSON file where questions are/will be saved.
                    Default: {directory_path}/questions.json
        batch_size: Recommended batch size for processing. Default: 3
    
    Returns:
        A status report including:
        - Total images found
        - List of processed images
        - List of pending images
        - Status of the output file (number of existing questions)
        - Recommendation for next steps
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
    
    # Determine output path
    if output_path:
        file_path = Path(output_path)
    else:
        file_path = dir_path / "questions.json"
        
    if file_path.suffix.lower() != ".json":
        file_path = file_path.with_suffix(".json")
        
    # Check existing questions and processed images
    existing_count = 0
    existing_info = "File does not exist"
    processed_images_set = set()
    
    if file_path.exists():
        questions, error = load_existing_questions(file_path)
        if not error:
            mc_count = len(questions.get("multiple_choice", []))
            tf_count = len(questions.get("true_false", []))
            existing_count = mc_count + tf_count
            existing_info = f"Contains {existing_count} questions ({mc_count} MC, {tf_count} TF)"
            
            # Get processed images
            # We try to handle both absolute and relative paths
            for p_str in questions.get("processed_images", []):
                p = Path(p_str)
                processed_images_set.add(str(p.resolve()))
                # Also add the raw string just in case
                processed_images_set.add(p_str)
        else:
            existing_info = f"Error reading file: {error}"

    # Determine pending images
    pending_images = []
    processed_count = 0
    
    # Map absolute paths to display paths
    found_images_map = {}
    for p_str in image_paths:
        p = Path(p_str)
        found_images_map[str(p.resolve())] = p_str
        
    for abs_p, display_p in found_images_map.items():
        # Check if absolute path or display path is in processed set
        if abs_p in processed_images_set or display_p in processed_images_set:
            processed_count += 1
        else:
            pending_images.append(display_p)
            
    # Sort pending images
    pending_images.sort()

    # Determine status
    if not pending_images and len(image_paths) > 0:
        status = "completed"
    elif processed_count > 0:
        status = "in_progress"
    else:
        status = "pending"

    # Create result object
    result = BatchProcessingResult(
        status=status,
        total_images=len(image_paths),
        total_questions=existing_count,
        successful_images=processed_count,
        processed_images=list(processed_images_set),
        unprocessed_images=pending_images,
        result_files=[str(file_path)] if file_path else [],
    )

    # Build status report
    lines = [
        "Batch Processing Status",
        "=" * 60,
        f"Directory: {directory_path}",
        f"Recursive: {'Yes' if recursive else 'No'}",
        f"Total Images Found: {result.total_images}",
        f"Processed Images: {len(result.processed_images)}",
        f"Pending Images: {len(result.unprocessed_images)}",
        "",
        f"Output File: {file_path}",
        f"Status: {existing_info}",
        "",
    ]
    
    lines.append(f"Next Batch to Process (Batch Size: {batch_size}):")
    
    # Only show the next batch of images to avoid overwhelming the context
    next_batch = result.unprocessed_images[:batch_size]
    for path in next_batch:
        lines.append(f"- {path}")
        
    if not result.unprocessed_images:
        lines.append("(None - All images processed!)")
        
    lines.append("")
    lines.append("Recommended Actions:")
    
    if result.unprocessed_images:
        lines.append("1. Call `analyze_image` with the images listed in 'Next Batch to Process' above.")
        lines.append("2. Repeat for remaining images.")
    else:
        lines.append("All images have been processed. You can review the output file.")
    
    return "\n".join(lines)



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
