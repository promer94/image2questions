"""
Base module for LangChain tools.

This module provides shared utilities and base classes for all tools.
"""

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standard result format for tool operations."""
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Human-readable message about the result")
    data: dict[str, Any] = Field(default_factory=dict, description="Additional result data")
    
    def __str__(self) -> str:
        """Return a string representation for the agent."""
        if self.success:
            result = f"✓ {self.message}"
        else:
            result = f"✗ {self.message}"
        
        if self.data:
            result += f"\nData: {self.data}"
        return result


class BatchProcessArgs(BaseModel):
    """Arguments for batch processing tool."""
    directory_path: str = Field(description="Path to the directory containing images")
    recursive: bool = Field(default=False, description="Whether to search subdirectories recursively")
    output_path: str | None = Field(default=None, description="Path to save the output JSON files")
    batch_size: int = Field(default=2, description="Number of images to process in each batch")


class ImageAnalysisResult(BaseModel):
    """Result from image analysis tool."""
    success: bool = Field(description="Whether analysis was successful")
    question_count: int = Field(default=0, description="Number of questions extracted")
    question_type: str = Field(default="", description="Type of questions (multiple_choice or true_false)")
    questions: list[dict] = Field(default_factory=list, description="List of extracted questions")
    source_images: list[str] = Field(default_factory=list, description="Source image paths")
    error: str | None = Field(default=None, description="Error message if failed")


class FileOperationResult(BaseModel):
    """Result from file generation tools."""
    success: bool = Field(description="Whether file operation was successful")
    file_path: str = Field(default="", description="Path to the generated file")
    items_processed: int = Field(default=0, description="Number of items processed")
    message: str = Field(default="", description="Additional information")
    error: str | None = Field(default=None, description="Error message if failed")


class ValidationResult(BaseModel):
    """Result from validation tool."""
    success: bool = Field(description="Whether validation passed")
    is_valid: bool = Field(description="Whether all questions are valid")
    total_questions: int = Field(default=0, description="Total questions validated")
    issues_count: int = Field(default=0, description="Number of issues found")
    confidence_score: float = Field(default=1.0, description="Overall confidence (0-1)")
    issues: list[dict] = Field(default_factory=list, description="List of validation issues")
    error: str | None = Field(default=None, description="Error message if failed")


class BatchProcessingResult(BaseModel):
    """Result from batch processing tool."""
    status: str = Field(description="Status of the batch processing (e.g., 'completed', 'in_progress', 'failed')")
    total_images: int = Field(default=0, description="Total images discovered")
    total_questions: int = Field(default=0, description="Total questions extracted and saved")
    successful_images: int = Field(default=0, description="Images processed successfully")
    failed_images: int = Field(default=0, description="Images that failed during processing")
    failed_image_paths: list[str] = Field(default_factory=list, description="Paths of images that failed during processing")
    processed_images: list[str] = Field(default_factory=list, description="All processed image paths")
    unprocessed_images: list[str] = Field(default_factory=list, description="Images not attempted or skipped")
    result_files: list[str] = Field(default_factory=list, description="JSON files created by the batch run")
    errors: list[str] = Field(default_factory=list, description="Error messages from failed images")
    error: str | None = Field(default=None, description="Overall error message if failed")
