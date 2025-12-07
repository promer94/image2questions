"""
Pydantic models for question data structures.

This module defines the data models for multiple-choice and true/false questions,
as well as extraction results and validation reports.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator


class QuestionType(str, Enum):
    """Enumeration of supported question types."""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    MIXED = "mixed"  # Both multiple choice and true/false
    AUTO = "auto"  # Auto-detect type


class MultipleChoiceOptions(BaseModel):
    """Options for a multiple-choice question."""
    a: str = Field(default="", description="Option A text")
    b: str = Field(default="", description="Option B text")
    c: str = Field(default="", description="Option C text")
    d: str = Field(default="", description="Option D text")

    def to_dict(self) -> dict[str, str]:
        """Convert options to a dictionary."""
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}

    def is_complete(self) -> bool:
        """Check if all options are non-empty."""
        return all([self.a, self.b, self.c, self.d])

    def non_empty_count(self) -> int:
        """Count non-empty options."""
        return sum(1 for opt in [self.a, self.b, self.c, self.d] if opt)


class QuestionBase(BaseModel):
    """Base class for all question types."""
    title: str = Field(description="The question text or statement")
    correct_answer: Optional[str] = Field(
        default=None,
        description="The correct answer (if known)"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata (difficulty, category, source, etc.)"
    )

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Validate that title is not empty."""
        if not v or not v.strip():
            raise ValueError("Question title cannot be empty")
        return v.strip()


class MultipleChoiceQuestion(QuestionBase):
    """A multiple-choice question with options A-D."""
    question_type: QuestionType = QuestionType.MULTIPLE_CHOICE
    options: MultipleChoiceOptions = Field(
        default_factory=MultipleChoiceOptions,
        description="The four options (A, B, C, D)"
    )

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, v: Optional[str]) -> Optional[str]:
        """Validate that correct answer is one of a, b, c, d."""
        if v is not None:
            v = v.lower().strip()
            if v not in ("a", "b", "c", "d"):
                raise ValueError("Correct answer must be one of: a, b, c, d")
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary format compatible with existing code."""
        return {
            "title": self.title,
            "options": self.options.to_dict(),
        }


class TrueFalseQuestion(QuestionBase):
    """A true/false question (判断题)."""
    question_type: QuestionType = QuestionType.TRUE_FALSE

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer(cls, v: Optional[str]) -> Optional[str]:
        """Validate that correct answer is true or false."""
        if v is not None:
            v = v.lower().strip()
            if v not in ("true", "false", "t", "f", "正确", "错误"):
                raise ValueError("Correct answer must be true/false")
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary format compatible with existing code."""
        return {
            "title": self.title,
        }


# Type alias for any question type
Question = Union[MultipleChoiceQuestion, TrueFalseQuestion]


class QuestionList(BaseModel):
    """A list of questions, used for LLM structured output."""
    questions: list[MultipleChoiceQuestion] = Field(
        default_factory=list,
        description="List of extracted questions"
    )


class TrueFalseList(BaseModel):
    """A list of true/false questions, used for LLM structured output."""
    questions: list[TrueFalseQuestion] = Field(
        default_factory=list,
        description="List of extracted true/false questions"
    )


class ValidationIssue(BaseModel):
    """A single validation issue for a question."""
    question_index: int = Field(description="Index of the question with the issue")
    issue_type: str = Field(description="Type of issue (e.g., 'empty_option', 'short_title')")
    message: str = Field(description="Human-readable description of the issue")
    severity: str = Field(
        default="warning",
        description="Severity level: 'error', 'warning', or 'info'"
    )


class ValidationReport(BaseModel):
    """Validation report for extracted questions."""
    is_valid: bool = Field(description="Whether all questions passed validation")
    total_questions: int = Field(description="Total number of questions validated")
    issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="List of validation issues found"
    )
    confidence_score: float = Field(
        default=1.0,
        description="Overall confidence score (0.0 to 1.0)"
    )

    def add_issue(
        self,
        question_index: int,
        issue_type: str,
        message: str,
        severity: str = "warning"
    ) -> None:
        """Add a validation issue to the report."""
        self.issues.append(ValidationIssue(
            question_index=question_index,
            issue_type=issue_type,
            message=message,
            severity=severity
        ))
        if severity == "error":
            self.is_valid = False


class ExtractionResult(BaseModel):
    """Result of a question extraction operation."""
    success: bool = Field(description="Whether extraction was successful")
    questions: list[Question] = Field(
        default_factory=list,
        description="List of extracted questions"
    )
    question_type: QuestionType = Field(
        default=QuestionType.AUTO,
        description="Type of questions extracted"
    )
    source_images: list[str] = Field(
        default_factory=list,
        description="List of source image paths"
    )
    validation: Optional[ValidationReport] = Field(
        default=None,
        description="Validation report for extracted questions"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if extraction failed"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of extraction"
    )

    @property
    def question_count(self) -> int:
        """Get the number of extracted questions."""
        return len(self.questions)


class AgentResponse(BaseModel):
    """Structured response format for the agent.
    
    This model ensures stable, predictable output from the agent,
    making it easier to parse and use in downstream applications.
    """
    success: bool = Field(
        description="Whether the operation was successful"
    )
    message: str = Field(
        description="Human-readable summary of what was done"
    )
    operation: str = Field(
        default="unknown",
        description="Type of operation performed (extract, save, validate, etc.)"
    )
    extracted_count: Optional[int] = Field(
        default=None,
        description="Number of questions extracted (if applicable)"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Output file path (if a file was created/modified)"
    )
    question_type: Optional[str] = Field(
        default=None,
        description="Type of questions processed (multiple_choice, true_false, mixed)"
    )
    details: Optional[dict] = Field(
        default=None,
        description="Additional details about the operation"
    )
