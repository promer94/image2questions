"""
Tests for Pydantic data models.

This module tests the question models, validation, and serialization.
"""

import pytest
from datetime import datetime

from src.models.questions import (
    MultipleChoiceOptions,
    MultipleChoiceQuestion,
    TrueFalseQuestion,
    QuestionType,
    QuestionList,
    TrueFalseList,
    ValidationIssue,
    ValidationReport,
    ExtractionResult,
)


class TestMultipleChoiceOptions:
    """Tests for MultipleChoiceOptions model."""
    
    def test_create_options(self):
        """Test creating options with all values."""
        options = MultipleChoiceOptions(
            a="Option A",
            b="Option B",
            c="Option C",
            d="Option D"
        )
        assert options.a == "Option A"
        assert options.b == "Option B"
        assert options.c == "Option C"
        assert options.d == "Option D"
    
    def test_default_empty_options(self):
        """Test that options default to empty strings."""
        options = MultipleChoiceOptions()
        assert options.a == ""
        assert options.b == ""
        assert options.c == ""
        assert options.d == ""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        options = MultipleChoiceOptions(a="A", b="B", c="C", d="D")
        result = options.to_dict()
        assert result == {"a": "A", "b": "B", "c": "C", "d": "D"}
    
    def test_is_complete(self):
        """Test completeness check."""
        complete = MultipleChoiceOptions(a="A", b="B", c="C", d="D")
        incomplete = MultipleChoiceOptions(a="A", b="B", c="", d="D")
        
        assert complete.is_complete() is True
        assert incomplete.is_complete() is False
    
    def test_non_empty_count(self):
        """Test counting non-empty options."""
        options = MultipleChoiceOptions(a="A", b="B", c="", d="")
        assert options.non_empty_count() == 2


class TestMultipleChoiceQuestion:
    """Tests for MultipleChoiceQuestion model."""
    
    def test_create_question(self, sample_multiple_choice_question):
        """Test creating a multiple-choice question."""
        question = MultipleChoiceQuestion(
            title=sample_multiple_choice_question["title"],
            options=MultipleChoiceOptions(**sample_multiple_choice_question["options"])
        )
        assert question.title == "What is the capital of France?"
        assert question.question_type == QuestionType.MULTIPLE_CHOICE
    
    def test_empty_title_raises_error(self):
        """Test that empty title raises validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            MultipleChoiceQuestion(title="", options=MultipleChoiceOptions())
    
    def test_whitespace_title_raises_error(self):
        """Test that whitespace-only title raises validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            MultipleChoiceQuestion(title="   ", options=MultipleChoiceOptions())
    
    def test_valid_correct_answer(self):
        """Test valid correct answer values."""
        for answer in ["a", "b", "c", "d", "A", "B", "C", "D"]:
            question = MultipleChoiceQuestion(
                title="Test question",
                options=MultipleChoiceOptions(),
                correct_answer=answer
            )
            assert question.correct_answer == answer.lower()
    
    def test_invalid_correct_answer(self):
        """Test invalid correct answer raises error."""
        with pytest.raises(ValueError, match="must be one of"):
            MultipleChoiceQuestion(
                title="Test question",
                options=MultipleChoiceOptions(),
                correct_answer="e"
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary format."""
        question = MultipleChoiceQuestion(
            title="Test question",
            options=MultipleChoiceOptions(a="A", b="B", c="C", d="D")
        )
        result = question.to_dict()
        assert result["title"] == "Test question"
        assert result["options"] == {"a": "A", "b": "B", "c": "C", "d": "D"}


class TestTrueFalseQuestion:
    """Tests for TrueFalseQuestion model."""
    
    def test_create_question(self, sample_true_false_question):
        """Test creating a true/false question."""
        question = TrueFalseQuestion(title=sample_true_false_question["title"])
        assert question.title == "The Earth is flat."
        assert question.question_type == QuestionType.TRUE_FALSE
    
    def test_valid_correct_answers(self):
        """Test valid correct answer values."""
        valid_answers = ["true", "false", "True", "False", "t", "f", "正确", "错误"]
        for answer in valid_answers:
            question = TrueFalseQuestion(
                title="Test statement",
                correct_answer=answer
            )
            assert question.correct_answer is not None
    
    def test_invalid_correct_answer(self):
        """Test invalid correct answer raises error."""
        with pytest.raises(ValueError, match="must be true/false"):
            TrueFalseQuestion(
                title="Test statement",
                correct_answer="maybe"
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary format."""
        question = TrueFalseQuestion(title="Test statement")
        result = question.to_dict()
        assert result == {"title": "Test statement"}


class TestQuestionList:
    """Tests for QuestionList model."""
    
    def test_create_empty_list(self):
        """Test creating an empty question list."""
        qlist = QuestionList()
        assert qlist.questions == []
    
    def test_create_with_questions(self):
        """Test creating a list with questions."""
        questions = [
            MultipleChoiceQuestion(
                title="Question 1",
                options=MultipleChoiceOptions(a="A", b="B", c="C", d="D")
            ),
            MultipleChoiceQuestion(
                title="Question 2",
                options=MultipleChoiceOptions(a="1", b="2", c="3", d="4")
            )
        ]
        qlist = QuestionList(questions=questions)
        assert len(qlist.questions) == 2


class TestValidationReport:
    """Tests for ValidationReport model."""
    
    def test_create_valid_report(self):
        """Test creating a validation report."""
        report = ValidationReport(
            is_valid=True,
            total_questions=5,
            confidence_score=0.95
        )
        assert report.is_valid is True
        assert report.total_questions == 5
        assert report.issues == []
    
    def test_add_issue(self):
        """Test adding validation issues."""
        report = ValidationReport(is_valid=True, total_questions=5)
        report.add_issue(
            question_index=0,
            issue_type="empty_option",
            message="Option C is empty",
            severity="warning"
        )
        
        assert len(report.issues) == 1
        assert report.issues[0].question_index == 0
        assert report.is_valid is True  # Warning doesn't invalidate
    
    def test_error_invalidates_report(self):
        """Test that error severity invalidates the report."""
        report = ValidationReport(is_valid=True, total_questions=5)
        report.add_issue(
            question_index=0,
            issue_type="missing_title",
            message="Question has no title",
            severity="error"
        )
        
        assert report.is_valid is False


class TestExtractionResult:
    """Tests for ExtractionResult model."""
    
    def test_successful_extraction(self):
        """Test creating a successful extraction result."""
        result = ExtractionResult(
            success=True,
            questions=[
                MultipleChoiceQuestion(
                    title="Test",
                    options=MultipleChoiceOptions(a="A", b="B", c="C", d="D")
                )
            ],
            source_images=["image1.jpg", "image2.jpg"]
        )
        
        assert result.success is True
        assert result.question_count == 1
        assert len(result.source_images) == 2
        assert isinstance(result.extracted_at, datetime)
    
    def test_failed_extraction(self):
        """Test creating a failed extraction result."""
        result = ExtractionResult(
            success=False,
            error_message="Failed to process image"
        )
        
        assert result.success is False
        assert result.question_count == 0
        assert result.error_message == "Failed to process image"
