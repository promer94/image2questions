"""
Unit tests for LangChain tools.

Tests cover the JSON generator, Word generator, validation, and batch processing tools.
Image analysis tests require mocking the LangChain ChatOpenAI client.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tools.json_generator import (
    parse_questions_input,
    load_existing_questions,
    save_questions_to_json,
    save_questions_json,
    load_questions_json,
)
from src.tools.validation import (
    validate_title,
    validate_multiple_choice_options,
    validate_multiple_choice_question,
    validate_true_false_question,
    validate_questions,
    calculate_confidence_score,
    validate_questions_tool,
)
from src.tools.batch_processor import (
    find_images_in_directory,
    SUPPORTED_EXTENSIONS,
)
from src.tools.image_analysis import (
    encode_image_to_base64,
    get_image_mime_type,
    validate_image_paths,
    build_image_content,
    extract_multiple_choice,
    extract_true_false,
    extract_mixed,
    analyze_image,
    Options,
    MultipleChoiceItem,
    MultipleChoiceResponse,
    TrueFalseItem,
    TrueFalseResponse,
    MixedResponse,
)


# ==================== JSON Generator Tests ====================

class TestParseQuestionsInput:
    """Tests for parse_questions_input function."""
    
    def test_valid_json_array(self):
        """Test that JSON array is rejected."""
        input_json = '[{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}]'
        questions, error = parse_questions_input(input_json)
        assert questions == {}
        assert "JSON object" in error
    
    def test_empty_array(self):
        """Test that empty array is rejected."""
        questions, error = parse_questions_input("[]")
        assert questions == {}
        assert "JSON object" in error
    
    def test_invalid_json(self):
        """Test parsing invalid JSON."""
        questions, error = parse_questions_input("not json")
        assert questions == {}
        assert "Invalid JSON" in error
    
    def test_json_object_instead_of_array(self):
        """Test that non-mixed JSON objects are rejected."""
        questions, error = parse_questions_input('{"title": "Q1"}')
        assert questions == {}
        assert "must contain 'multiple_choice' or 'true_false' keys" in error
    
    def test_mixed_format(self):
        """Test parsing mixed format from image analysis."""
        input_json = '{"multiple_choice": [{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}], "true_false": [{"title": "Q2"}]}'
        data, error = parse_questions_input(input_json)
        assert error is None
        assert isinstance(data, dict)
        assert len(data["multiple_choice"]) == 1
        assert len(data["true_false"]) == 1
    
    def test_mixed_format_only_multiple_choice(self):
        """Test parsing mixed format with only multiple choice."""
        input_json = '{"multiple_choice": [{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}]}'
        data, error = parse_questions_input(input_json)
        assert error is None
        assert isinstance(data, dict)
        assert len(data["multiple_choice"]) == 1
    
    def test_mixed_format_only_true_false(self):
        """Test parsing mixed format with only true/false."""
        input_json = '{"true_false": [{"title": "Q1"}]}'
        data, error = parse_questions_input(input_json)
        assert error is None
        assert isinstance(data, dict)
        assert len(data["true_false"]) == 1


class TestSaveQuestionsToJson:
    """Tests for save_questions_to_json function."""
    
    def test_save_questions(self, tmp_path):
        """Test saving questions to a JSON file."""
        questions = {
            "multiple_choice": [{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}],
            "true_false": []
        }
        output_path = tmp_path / "output.json"
        
        success, message = save_questions_to_json(questions, output_path, pretty=True)
        
        assert success
        assert output_path.exists()
        
        # Verify content
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(content["multiple_choice"]) == 1
        assert content["multiple_choice"][0]["title"] == "Q1"
    
    def test_save_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        questions = {"multiple_choice": [{"title": "Q1"}], "true_false": []}
        output_path = tmp_path / "subdir" / "nested" / "output.json"
        
        success, message = save_questions_to_json(questions, output_path)
        
        assert success
        assert output_path.exists()
    
    def test_save_with_chinese_characters(self, tmp_path):
        """Test saving questions with Chinese characters."""
        questions = {"multiple_choice": [{"title": "这是一道选择题"}], "true_false": []}
        output_path = tmp_path / "chinese.json"
        
        success, message = save_questions_to_json(questions, output_path)
        
        assert success
        content = output_path.read_text(encoding="utf-8")
        assert "这是一道选择题" in content


class TestLoadExistingQuestions:
    """Tests for load_existing_questions function."""
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from a non-existent file."""
        questions, error = load_existing_questions(tmp_path / "nonexistent.json")
        assert questions == {}
        assert error is None
    
    def test_load_valid_file(self, tmp_path):
        """Test loading from a valid JSON file."""
        file_path = tmp_path / "questions.json"
        file_path.write_text('{"multiple_choice": [{"title": "Q1"}], "true_false": []}', encoding="utf-8")
        
        questions, error = load_existing_questions(file_path)
        
        assert error is None
        assert len(questions["multiple_choice"]) == 1
        assert questions["multiple_choice"][0]["title"] == "Q1"
    
    def test_load_invalid_json(self, tmp_path):
        """Test loading from a file with invalid JSON."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json", encoding="utf-8")
        
        questions, error = load_existing_questions(file_path)
        
        assert questions == {}
        assert "invalid JSON" in error


class TestSaveQuestionsJsonTool:
    """Tests for the save_questions_json tool."""
    
    def test_save_multiple_choice(self, tmp_path):
        """Test saving multiple choice questions."""
        questions = {
            "multiple_choice": [{"title": "What is 2+2?", "options": {"a": "3", "b": "4", "c": "5", "d": "6"}}],
            "true_false": []
        }
        output_path = tmp_path / "mc.json"
        
        result = save_questions_json.invoke({
            "questions_json": json.dumps(questions),
            "output_path": str(output_path)
        })
        
        assert "Saved 1 questions" in result
        assert output_path.exists()
    
    def test_save_true_false(self, tmp_path):
        """Test saving true/false questions."""
        questions = {
            "multiple_choice": [],
            "true_false": [{"title": "The sky is blue."}]
        }
        output_path = tmp_path / "tf.json"
        
        result = save_questions_json.invoke({
            "questions_json": json.dumps(questions),
            "output_path": str(output_path)
        })
        
        assert "Saved 1 questions" in result
    
    def test_append_mode(self, tmp_path):
        """Test appending to existing file."""
        output_path = tmp_path / "append.json"
        
        # Save initial questions
        initial = {"multiple_choice": [{"title": "Q1"}], "true_false": []}
        output_path.write_text(json.dumps(initial), encoding="utf-8")
        
        # Append more questions
        additional = {"multiple_choice": [{"title": "Q2"}], "true_false": []}
        result = save_questions_json.invoke({
            "questions_json": json.dumps(additional),
            "output_path": str(output_path),
            "append": True
        })
        
        assert "Appended" in result
        
        # Verify combined content
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(content["multiple_choice"]) == 2
    
    def test_error_invalid_json(self, tmp_path):
        """Test error handling for invalid JSON."""
        result = save_questions_json.invoke({
            "questions_json": "not json",
            "output_path": str(tmp_path / "out.json")
        })
        
        assert "Error" in result
    
    def test_error_empty_questions(self, tmp_path):
        """Test error handling for empty questions."""
        result = save_questions_json.invoke({
            "questions_json": json.dumps({"multiple_choice": [], "true_false": []}),
            "output_path": str(tmp_path / "out.json")
        })
        
        assert "Error" in result
        assert "empty" in result.lower()
    
    def test_save_mixed_format(self, tmp_path):
        """Test saving mixed format from image analysis."""
        mixed_data = {
            "multiple_choice": [{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}],
            "true_false": [{"title": "Q2"}]
        }
        output_path = tmp_path / "mixed.json"
        
        result = save_questions_json.invoke({
            "questions_json": json.dumps(mixed_data),
            "output_path": str(output_path)
        })
        
        assert "Saved 2 questions" in result
        assert output_path.exists()
        
        # Verify content preserves mixed format
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert "multiple_choice" in content
        assert "true_false" in content
    
    def test_save_mixed_format_extract_multiple_choice(self, tmp_path):
        """Test extracting only multiple choice from mixed format."""
        mixed_data = {
            "multiple_choice": [{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}],
            "true_false": [{"title": "Q2"}]
        }
        output_path = tmp_path / "mc_only.json"
        
        result = save_questions_json.invoke({
            "questions_json": json.dumps(mixed_data),
            "output_path": str(output_path),
            "question_type": "multiple_choice"
        })
        
        assert "Saved 1 questions" in result
        
        # Verify content is only multiple choice (as dict with empty true_false)
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(content, dict)
        assert len(content["multiple_choice"]) == 1
        assert len(content["true_false"]) == 0
        assert "options" in content["multiple_choice"][0]
    
    def test_save_mixed_format_extract_true_false(self, tmp_path):
        """Test extracting only true/false from mixed format."""
        mixed_data = {
            "multiple_choice": [{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}],
            "true_false": [{"title": "Q2"}, {"title": "Q3"}]
        }
        output_path = tmp_path / "tf_only.json"
        
        result = save_questions_json.invoke({
            "questions_json": json.dumps(mixed_data),
            "output_path": str(output_path),
            "question_type": "true_false"
        })
        
        assert "Saved 2 questions" in result
        
        # Verify content is only true/false (as dict with empty multiple_choice)
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(content, dict)
        assert len(content["true_false"]) == 2
        assert len(content["multiple_choice"]) == 0
        assert "options" not in content["true_false"][0]

    def test_save_with_processed_images(self, tmp_path):
        """Test saving questions with processed images list."""
        questions = {
            "multiple_choice": [{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}],
            "true_false": []
        }
        output_path = tmp_path / "processed.json"
        processed_images = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
        
        result = save_questions_json.invoke({
            "questions_json": json.dumps(questions),
            "output_path": str(output_path),
            "processed_images": processed_images
        })
        
        assert "Saved 1 questions" in result
        
        # Verify content
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert "processed_images" in content
        assert len(content["processed_images"]) == 2
        assert "/path/to/img1.jpg" in content["processed_images"]

    def test_append_with_processed_images(self, tmp_path):
        """Test appending questions and merging processed images."""
        output_path = tmp_path / "append_processed.json"
        
        # Initial save
        initial = {
            "multiple_choice": [{"title": "Q1"}], 
            "true_false": [],
            "processed_images": ["img1.jpg"]
        }
        output_path.write_text(json.dumps(initial), encoding="utf-8")
        
        # Append
        additional = {"multiple_choice": [{"title": "Q2"}], "true_false": []}
        new_images = ["img2.jpg", "img1.jpg"] # img1.jpg is duplicate
        
        result = save_questions_json.invoke({
            "questions_json": json.dumps(additional),
            "output_path": str(output_path),
            "append": True,
            "processed_images": new_images
        })
        
        assert "Appended" in result
        
        # Verify content
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(content["multiple_choice"]) == 2
        assert len(content["processed_images"]) == 2 # Should be unique
        assert "img1.jpg" in content["processed_images"]
        assert "img2.jpg" in content["processed_images"]


class TestLoadQuestionsJsonTool:
    """Tests for the load_questions_json tool."""
    
    def test_load_valid_file(self, tmp_path):
        """Test loading a valid JSON file."""
        file_path = tmp_path / "questions.json"
        questions = {"multiple_choice": [{"title": "Q1"}, {"title": "Q2"}], "true_false": []}
        file_path.write_text(json.dumps(questions), encoding="utf-8")
        
        result = load_questions_json.invoke({"file_path": str(file_path)})
        
        assert "Loaded 2 questions" in result
        assert "Q1" in result
        assert "Q2" in result
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a non-existent file."""
        result = load_questions_json.invoke({
            "file_path": str(tmp_path / "nonexistent.json")
        })
        
        assert "Error" in result
        assert "not found" in result.lower()


# ==================== Validation Tests ====================

class TestValidateTitle:
    """Tests for validate_title function."""
    
    def test_valid_title(self):
        """Test a valid title."""
        issues = validate_title("What is the capital of France?", 0)
        assert len(issues) == 0
    
    def test_empty_title(self):
        """Test an empty title."""
        issues = validate_title("", 0)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "empty_title"
        assert issues[0]["severity"] == "error"
    
    def test_short_title(self):
        """Test a very short title."""
        issues = validate_title("Hi", 0)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "short_title"
        assert issues[0]["severity"] == "warning"


class TestValidateMultipleChoiceOptions:
    """Tests for validate_multiple_choice_options function."""
    
    def test_valid_options(self):
        """Test valid options."""
        options = {"a": "Option A", "b": "Option B", "c": "Option C", "d": "Option D"}
        issues = validate_multiple_choice_options(options, 0)
        assert len(issues) == 0
    
    def test_missing_options(self):
        """Test missing options."""
        options = {"a": "Option A", "b": "Option B"}
        issues = validate_multiple_choice_options(options, 0)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "missing_options"
    
    def test_empty_options(self):
        """Test empty options."""
        options = {"a": "Option A", "b": "", "c": "Option C", "d": ""}
        issues = validate_multiple_choice_options(options, 0)
        assert any(i["issue_type"] == "empty_options" for i in issues)
    
    def test_all_empty_options(self):
        """Test all empty options."""
        options = {"a": "", "b": "", "c": "", "d": ""}
        issues = validate_multiple_choice_options(options, 0)
        assert any(i["issue_type"] == "all_empty_options" for i in issues)
    
    def test_duplicate_options(self):
        """Test duplicate options."""
        options = {"a": "Same", "b": "Same", "c": "Different", "d": "Also Different"}
        issues = validate_multiple_choice_options(options, 0)
        assert any(i["issue_type"] == "duplicate_options" for i in issues)


class TestValidateQuestions:
    """Tests for validate_questions function."""
    
    def test_valid_multiple_choice(self):
        """Test validating valid multiple choice questions."""
        questions = [
            {
                "title": "What is 2+2?",
                "options": {"a": "3", "b": "4", "c": "5", "d": "6"}
            }
        ]
        report = validate_questions(questions, "multiple_choice")
        
        assert report["is_valid"]
        assert report["total_questions"] == 1
        assert report["error_count"] == 0
    
    def test_invalid_multiple_choice(self):
        """Test validating invalid multiple choice questions."""
        questions = [
            {
                "title": "",  # Empty title
                "options": {"a": "A", "b": "B", "c": "C", "d": "D"}
            }
        ]
        report = validate_questions(questions, "multiple_choice")
        
        assert not report["is_valid"]
        assert report["error_count"] > 0
    
    def test_valid_true_false(self):
        """Test validating valid true/false questions."""
        questions = [{"title": "The Earth is round."}]
        report = validate_questions(questions, "true_false")
        
        assert report["is_valid"]
        assert report["total_questions"] == 1


class TestCalculateConfidenceScore:
    """Tests for calculate_confidence_score function."""
    
    def test_perfect_score(self):
        """Test confidence score with no issues."""
        questions = [{"title": "Q1"}]
        issues = []
        score = calculate_confidence_score(questions, issues)
        assert score == 1.0
    
    def test_score_with_errors(self):
        """Test confidence score with errors."""
        questions = [{"title": "Q1"}]
        issues = [{"severity": "error"}]
        score = calculate_confidence_score(questions, issues)
        assert score < 1.0
    
    def test_empty_questions(self):
        """Test confidence score with no questions."""
        score = calculate_confidence_score([], [])
        assert score == 0.0


class TestValidateQuestionsTool:
    """Tests for the validate_questions_tool tool."""
    
    def test_validate_valid_questions(self):
        """Test validating valid questions."""
        questions = [
            {
                "title": "What is the capital of France?",
                "options": {"a": "London", "b": "Paris", "c": "Berlin", "d": "Madrid"}
            }
        ]
        
        result = validate_questions_tool.invoke({
            "questions_json": json.dumps(questions),
            "question_type": "multiple_choice"
        })
        
        assert "VALID" in result
        assert "Confidence Score" in result
    
    def test_validate_invalid_questions(self):
        """Test validating invalid questions."""
        questions = [{"title": "", "options": {"a": "", "b": "", "c": "", "d": ""}}]
        
        result = validate_questions_tool.invoke({
            "questions_json": json.dumps(questions),
            "question_type": "multiple_choice"
        })
        
        assert "INVALID" in result
    
    def test_error_invalid_json(self):
        """Test error handling for invalid JSON."""
        result = validate_questions_tool.invoke({
            "questions_json": "not json",
            "question_type": "multiple_choice"
        })
        
        assert "Error" in result


# ==================== Batch Processor Tests ====================

class TestFindImagesInDirectory:
    """Tests for find_images_in_directory function."""
    
    def test_find_images(self, tmp_path):
        """Test finding images in a directory."""
        # Create test images
        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.png").touch()
        (tmp_path / "document.txt").touch()
        
        images = find_images_in_directory(tmp_path)
        
        assert len(images) == 2
        assert any("image1.jpg" in p for p in images)
        assert any("image2.png" in p for p in images)
    
    def test_find_images_recursive(self, tmp_path):
        """Test finding images recursively."""
        # Create test structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "image1.jpg").touch()
        (subdir / "image2.png").touch()
        
        # Non-recursive
        images_flat = find_images_in_directory(tmp_path, recursive=False)
        assert len(images_flat) == 1
        
        # Recursive
        images_recursive = find_images_in_directory(tmp_path, recursive=True)
        assert len(images_recursive) == 2
    
    def test_empty_directory(self, tmp_path):
        """Test with an empty directory."""
        images = find_images_in_directory(tmp_path)
        assert images == []
    
    def test_case_insensitive_extensions(self, tmp_path):
        """Test that extensions are case-insensitive."""
        (tmp_path / "image1.JPG").touch()
        (tmp_path / "image2.PNG").touch()
        
        images = find_images_in_directory(tmp_path)
        
        assert len(images) == 2


# ==================== Word Generator Tests ====================

class TestWordGenerator:
    """Tests for Word generator functions."""
    
    def test_word_generator_import(self):
        """Test that Word generator can be imported."""
        from src.tools.word_generator import (
            save_questions_word,
            generate_word_document,
        )
        assert save_questions_word is not None
        assert generate_word_document is not None
    
    def test_generate_multiple_choice_document(self, tmp_path):
        """Test generating a multiple choice Word document."""
        from src.tools.word_generator import generate_word_document
        
        questions = [
            {
                "title": "What is 2+2?",
                "options": {"a": "3", "b": "4", "c": "5", "d": "6"}
            }
        ]
        output_path = tmp_path / "mc.docx"
        
        success, message, count = generate_word_document(
            questions, output_path, "multiple_choice"
        )
        
        assert success
        assert count == 1
        assert output_path.exists()
    
    def test_generate_true_false_document(self, tmp_path):
        """Test generating a true/false Word document."""
        from src.tools.word_generator import generate_word_document
        
        questions = [{"title": "The sky is blue."}]
        output_path = tmp_path / "tf.docx"
        
        success, message, count = generate_word_document(
            questions, output_path, "true_false"
        )
        
        assert success
        assert count == 1
        assert output_path.exists()
    
    def test_save_questions_word_tool(self, tmp_path):
        """Test the save_questions_word tool."""
        from src.tools.word_generator import save_questions_word
        
        questions = [
            {
                "title": "What is 2+2?",
                "options": {"a": "3", "b": "4", "c": "5", "d": "6"}
            }
        ]
        output_path = tmp_path / "output.docx"
        
        result = save_questions_word.invoke({
            "questions_json": json.dumps(questions),
            "output_path": str(output_path),
            "question_type": "multiple_choice"
        })
        
        assert "Successfully" in result or "Created" in result
        assert output_path.exists()
    
    def test_save_questions_word_mixed_format(self, tmp_path):
        """Test the save_questions_word tool with mixed format from image analysis."""
        from src.tools.word_generator import save_questions_word
        
        mixed_data = {
            "multiple_choice": [
                {"title": "What is 2+2?", "options": {"a": "3", "b": "4", "c": "5", "d": "6"}}
            ],
            "true_false": [
                {"title": "The sky is blue."}
            ]
        }
        output_path = tmp_path / "mixed.docx"
        
        result = save_questions_word.invoke({
            "questions_json": json.dumps(mixed_data),
            "output_path": str(output_path),
            "question_type": "auto"
        })
        
        assert "Successfully" in result or "Created" in result
        assert output_path.exists()
        assert "2 questions" in result
        assert "1 multiple choice" in result
        assert "1 true/false" in result
    
    def test_save_questions_word_mixed_extract_mc(self, tmp_path):
        """Test extracting only multiple choice from mixed format."""
        from src.tools.word_generator import save_questions_word
        
        mixed_data = {
            "multiple_choice": [
                {"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}
            ],
            "true_false": [
                {"title": "Q2"}
            ]
        }
        output_path = tmp_path / "mc_only.docx"
        
        result = save_questions_word.invoke({
            "questions_json": json.dumps(mixed_data),
            "output_path": str(output_path),
            "question_type": "multiple_choice"
        })
        
        assert "Successfully" in result or "Created" in result
        assert output_path.exists()
    
    def test_save_questions_word_mixed_extract_tf(self, tmp_path):
        """Test extracting only true/false from mixed format."""
        from src.tools.word_generator import save_questions_word
        
        mixed_data = {
            "multiple_choice": [
                {"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}
            ],
            "true_false": [
                {"title": "Q2"}
            ]
        }
        output_path = tmp_path / "tf_only.docx"
        
        result = save_questions_word.invoke({
            "questions_json": json.dumps(mixed_data),
            "output_path": str(output_path),
            "question_type": "true_false"
        })
        
        assert "Successfully" in result or "Created" in result
        assert output_path.exists()
    
    def test_save_questions_word_auto_detect(self, tmp_path):
        """Test auto-detecting question type from list format."""
        from src.tools.word_generator import save_questions_word
        
        # Multiple choice with options
        mc_questions = [{"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}}]
        output_path = tmp_path / "auto_mc.docx"
        
        result = save_questions_word.invoke({
            "questions_json": json.dumps(mc_questions),
            "output_path": str(output_path),
            "question_type": "auto"
        })
        
        assert "Successfully" in result or "Created" in result
        
        # True/false without options
        tf_questions = [{"title": "The earth is round."}]
        output_path2 = tmp_path / "auto_tf.docx"
        
        result2 = save_questions_word.invoke({
            "questions_json": json.dumps(tf_questions),
            "output_path": str(output_path2),
            "question_type": "auto"
        })
        
        assert "Successfully" in result2 or "Created" in result2


# ==================== Integration Tests ====================

class TestToolsIntegration:
    """Integration tests for the tools module."""
    
    def test_all_tools_importable(self):
        """Test that all tools can be imported."""
        from src.tools import (
            analyze_image,
            save_questions_json,
            load_questions_json,
            save_questions_word,
            validate_questions_tool,
            batch_process_images,
            get_all_tools,
        )
        
        tools = get_all_tools()
        assert len(tools) == 6
    
    def test_tool_has_name_and_description(self):
        """Test that tools have proper names and descriptions."""
        from src.tools import get_all_tools
        
        for tool in get_all_tools():
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert tool.name
            assert tool.description
    
    def test_workflow_json_save_and_load(self, tmp_path):
        """Test a complete workflow: save and load JSON."""
        from src.tools import save_questions_json, load_questions_json
        
        questions = {
            "multiple_choice": [
                {"title": "Q1", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}},
                {"title": "Q2", "options": {"a": "A", "b": "B", "c": "C", "d": "D"}},
            ],
            "true_false": []
        }
        file_path = tmp_path / "workflow.json"
        
        # Save
        save_result = save_questions_json.invoke({
            "questions_json": json.dumps(questions),
            "output_path": str(file_path)
        })
        assert "Saved 2 questions" in save_result
        
        # Load
        load_result = load_questions_json.invoke({
            "file_path": str(file_path)
        })
        assert "Loaded 2 questions" in load_result
        assert "Q1" in load_result
        assert "Q2" in load_result
    
    def test_workflow_validate_and_save(self, tmp_path):
        """Test workflow: validate then save."""
        from src.tools import validate_questions_tool, save_questions_json
        
        questions = {
            "multiple_choice": [
                {"title": "What is the capital of France?", 
                 "options": {"a": "London", "b": "Paris", "c": "Berlin", "d": "Madrid"}}
            ],
            "true_false": []
        }
        
        # Validate
        validate_result = validate_questions_tool.invoke({
            "questions_json": json.dumps(questions),
            "question_type": "multiple_choice"
        })
        assert "VALID" in validate_result
        
        # Save
        file_path = tmp_path / "validated.json"
        save_result = save_questions_json.invoke({
            "questions_json": json.dumps(questions),
            "output_path": str(file_path)
        })
        assert "Saved" in save_result


# ==================== Image Analysis Tests ====================

class TestImageAnalysisHelpers:
    """Tests for image analysis helper functions."""
    
    def test_get_image_mime_type_jpg(self):
        """Test MIME type for JPG files."""
        assert get_image_mime_type("test.jpg") == "image/jpeg"
        assert get_image_mime_type("test.jpeg") == "image/jpeg"
    
    def test_get_image_mime_type_png(self):
        """Test MIME type for PNG files."""
        assert get_image_mime_type("test.png") == "image/png"
    
    def test_get_image_mime_type_other(self):
        """Test MIME type for other formats."""
        assert get_image_mime_type("test.gif") == "image/gif"
        assert get_image_mime_type("test.webp") == "image/webp"
        assert get_image_mime_type("test.bmp") == "image/bmp"
    
    def test_get_image_mime_type_unknown(self):
        """Test MIME type for unknown extension defaults to jpeg."""
        assert get_image_mime_type("test.unknown") == "image/jpeg"
    
    def test_validate_image_paths_nonexistent(self, tmp_path):
        """Test validation of non-existent paths."""
        paths = [str(tmp_path / "nonexistent.png")]
        valid, errors = validate_image_paths(paths)
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert "not found" in errors[0].lower()
    
    def test_validate_image_paths_valid(self, tmp_path):
        """Test validation of valid paths."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image data")
        
        valid, errors = validate_image_paths([str(image_path)])
        
        assert len(valid) == 1
        assert len(errors) == 0
    
    def test_validate_image_paths_directory(self, tmp_path):
        """Test that directories are rejected."""
        valid, errors = validate_image_paths([str(tmp_path)])
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert "Not a file" in errors[0]
    
    def test_validate_image_paths_mixed(self, tmp_path):
        """Test mixed valid and invalid paths."""
        valid_path = tmp_path / "valid.png"
        valid_path.write_bytes(b"fake image")
        invalid_path = tmp_path / "nonexistent.png"
        
        valid, errors = validate_image_paths([str(valid_path), str(invalid_path)])
        
        assert len(valid) == 1
        assert len(errors) == 1
    
    def test_encode_image_to_base64(self, tmp_path):
        """Test base64 encoding of image."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"test image data")
        
        encoded = encode_image_to_base64(str(image_path))
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Verify it's valid base64
        import base64
        decoded = base64.standard_b64decode(encoded)
        assert decoded == b"test image data"
    
    def test_build_image_content(self, tmp_path):
        """Test building image content for API."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"test image data")
        
        content = build_image_content([str(image_path)])
        
        assert len(content) == 1
        assert content[0]["type"] == "input_image"
        assert "data:image/png;base64," in content[0]["image_url"]


class TestImageAnalysisPydanticModels:
    """Tests for Pydantic models used in image analysis."""
    
    def test_options_model(self):
        """Test Options model."""
        options = Options(a="A", b="B", c="C", d="D")
        assert options.a == "A"
        assert options.b == "B"
        assert options.c == "C"
        assert options.d == "D"
    
    def test_options_model_defaults(self):
        """Test Options model with defaults."""
        options = Options()
        assert options.a == ""
        assert options.b == ""
        assert options.c == ""
        assert options.d == ""
    
    def test_multiple_choice_item(self):
        """Test MultipleChoiceItem model."""
        item = MultipleChoiceItem(
            title="What is 2+2?",
            options=Options(a="3", b="4", c="5", d="6")
        )
        assert item.title == "What is 2+2?"
        assert item.options.b == "4"
    
    def test_multiple_choice_response(self):
        """Test MultipleChoiceResponse model."""
        response = MultipleChoiceResponse(questions=[
            MultipleChoiceItem(title="Q1", options=Options(a="A", b="B", c="C", d="D")),
            MultipleChoiceItem(title="Q2", options=Options(a="1", b="2", c="3", d="4")),
        ])
        assert len(response.questions) == 2
        assert response.questions[0].title == "Q1"
    
    def test_true_false_item(self):
        """Test TrueFalseItem model."""
        item = TrueFalseItem(title="The sky is blue.")
        assert item.title == "The sky is blue."
    
    def test_true_false_response(self):
        """Test TrueFalseResponse model."""
        response = TrueFalseResponse(questions=[
            TrueFalseItem(title="Statement 1"),
            TrueFalseItem(title="Statement 2"),
        ])
        assert len(response.questions) == 2
    
    def test_mixed_response(self):
        """Test MixedResponse model."""
        response = MixedResponse(
            multiple_choice_questions=[
                MultipleChoiceItem(title="MC Q1", options=Options(a="A", b="B", c="C", d="D")),
            ],
            true_false_questions=[
                TrueFalseItem(title="TF Q1"),
                TrueFalseItem(title="TF Q2"),
            ]
        )
        assert len(response.multiple_choice_questions) == 1
        assert len(response.true_false_questions) == 2
    
    def test_mixed_response_empty(self):
        """Test MixedResponse model with empty lists."""
        response = MixedResponse()
        assert len(response.multiple_choice_questions) == 0
        assert len(response.true_false_questions) == 0


class TestAnalyzeImageTool:
    """Tests for the analyze_image tool."""
    
    def test_no_image_paths(self):
        """Test error when no image paths provided."""
        result = analyze_image.invoke({
            "image_paths": "",
            "question_type": "multiple_choice"
        })
        assert "Error" in result
        assert "No image paths provided" in result
    
    def test_invalid_question_type(self, tmp_path):
        """Test error for invalid question type."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image")
        
        result = analyze_image.invoke({
            "image_paths": str(image_path),
            "question_type": "invalid_type"
        })
        assert "Error" in result
        assert "Invalid question_type" in result
    
    def test_nonexistent_image_path(self, tmp_path):
        """Test error for non-existent image."""
        result = analyze_image.invoke({
            "image_paths": str(tmp_path / "nonexistent.png"),
            "question_type": "multiple_choice"
        })
        assert "Error" in result
        assert "No valid images" in result
    
    def test_valid_question_types_accepted(self, tmp_path):
        """Test that all valid question types are accepted (before API call)."""
        # We just verify the validation passes, not the actual API call
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image")
        
        # These should not fail with "Invalid question_type"
        for qtype in ["multiple_choice", "true_false", "mixed"]:
            result = analyze_image.invoke({
                "image_paths": str(image_path),
                "question_type": qtype
            })
            assert "Invalid question_type" not in result


class TestExtractFunctions:
    """Tests for extract functions with mocked LangChain create_agent."""
    
    def test_extract_multiple_choice(self):
        """Test extract_multiple_choice with mocked create_agent."""
        mock_llm = MagicMock()
        mock_agent = MagicMock()
        
        # Mock the agent invoke response
        mock_agent.invoke.return_value = {
            "structured_response": MultipleChoiceResponse(
                questions=[
                    MultipleChoiceItem(
                        title="What is 2+2?",
                        options=Options(a="3", b="4", c="5", d="6")
                    ),
                    MultipleChoiceItem(
                        title="Capital of France?",
                        options=Options(a="London", b="Paris", c="Berlin", d="Madrid")
                    ),
                ]
            )
        }
        
        with patch("src.tools.image_analysis.build_image_content", return_value=[]):
            with patch("src.tools.image_analysis.create_agent", return_value=mock_agent):
                result = extract_multiple_choice(mock_llm, ["fake_path.png"])
        
        assert result["type"] == "multiple_choice"
        assert len(result["multiple_choice"]) == 2
        assert result["multiple_choice"][0]["title"] == "What is 2+2?"
        assert result["multiple_choice"][0]["options"]["a"] == "3"
        assert result["multiple_choice"][0]["options"]["b"] == "4"
        assert result["multiple_choice"][0]["source_image"] == ["fake_path.png"]
        assert result["multiple_choice"][1]["title"] == "Capital of France?"
        assert result["multiple_choice"][1]["source_image"] == ["fake_path.png"]
    
    def test_extract_true_false(self):
        """Test extract_true_false with mocked create_agent."""
        mock_llm = MagicMock()
        mock_agent = MagicMock()
        
        # Mock the agent invoke response
        mock_agent.invoke.return_value = {
            "structured_response": TrueFalseResponse(
                questions=[
                    TrueFalseItem(title="The sky is blue."),
                    TrueFalseItem(title="Water boils at 50°C."),
                ]
            )
        }
        
        with patch("src.tools.image_analysis.build_image_content", return_value=[]):
            with patch("src.tools.image_analysis.create_agent", return_value=mock_agent):
                result = extract_true_false(mock_llm, ["fake_path.png"])
        
        assert result["type"] == "true_false"
        assert len(result["true_false"]) == 2
        assert result["true_false"][0]["title"] == "The sky is blue."
        assert result["true_false"][0]["source_image"] == ["fake_path.png"]
        assert result["true_false"][1]["title"] == "Water boils at 50°C."
        assert result["true_false"][1]["source_image"] == ["fake_path.png"]
    
    def test_extract_mixed(self):
        """Test extract_mixed with mocked create_agent."""
        mock_llm = MagicMock()
        mock_agent = MagicMock()
        
        # Mock the agent invoke response
        mock_agent.invoke.return_value = {
            "structured_response": MixedResponse(
                multiple_choice_questions=[
                    MultipleChoiceItem(
                        title="What is 2+2?",
                        options=Options(a="3", b="4", c="5", d="6")
                    ),
                ],
                true_false_questions=[
                    TrueFalseItem(title="The sky is blue."),
                    TrueFalseItem(title="Fire is cold."),
                ]
            )
        }
        
        with patch("src.tools.image_analysis.build_image_content", return_value=[]):
            with patch("src.tools.image_analysis.create_agent", return_value=mock_agent):
                result = extract_mixed(mock_llm, ["fake_path.png"])
        
        assert result["type"] == "mixed"
        assert "multiple_choice" in result
        assert "true_false" in result
        assert len(result["multiple_choice"]) == 1
        assert len(result["true_false"]) == 2
        assert result["multiple_choice"][0]["title"] == "What is 2+2?"
        assert result["multiple_choice"][0]["options"]["b"] == "4"
        assert result["multiple_choice"][0]["source_image"] == ["fake_path.png"]
        assert result["true_false"][0]["title"] == "The sky is blue."
        assert result["true_false"][0]["source_image"] == ["fake_path.png"]
        assert result["true_false"][1]["title"] == "Fire is cold."
        assert result["true_false"][1]["source_image"] == ["fake_path.png"]
    
    def test_extract_mixed_empty_one_type(self):
        """Test extract_mixed when one type has no questions."""
        mock_llm = MagicMock()
        mock_agent = MagicMock()
        
        # Mock the agent invoke response
        mock_agent.invoke.return_value = {
            "structured_response": MixedResponse(
                multiple_choice_questions=[],
                true_false_questions=[
                    TrueFalseItem(title="Statement 1"),
                ]
            )
        }
        
        with patch("src.tools.image_analysis.build_image_content", return_value=[]):
            with patch("src.tools.image_analysis.create_agent", return_value=mock_agent):
                result = extract_mixed(mock_llm, ["fake_path.png"])
        
        assert result["type"] == "mixed"
        assert len(result["multiple_choice"]) == 0
        assert len(result["true_false"]) == 1


class TestAnalyzeImageToolWithMocking:
    """Integration tests for analyze_image tool with mocked dependencies."""
    
    @patch("src.tools.image_analysis.create_agent")
    @patch("src.tools.image_analysis.ChatOpenAI")
    @patch("src.tools.image_analysis.get_settings")
    def test_analyze_multiple_choice_success(self, mock_get_settings, mock_chat_openai_class, mock_create_agent, tmp_path):
        """Test successful multiple choice extraction."""
        # Setup mock settings
        mock_settings = MagicMock()
        mock_settings.doubao_api_key = "test-key"
        mock_settings.doubao_base_url = "https://test.api"
        mock_settings.doubao_model = "test-model"
        mock_settings.doubao_max_tokens = 4096
        mock_get_settings.return_value = mock_settings
        
        # Setup mock LangChain ChatOpenAI client
        mock_llm = MagicMock()
        mock_chat_openai_class.return_value = mock_llm
        
        # Setup mock agent
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "structured_response": MultipleChoiceResponse(
                questions=[
                    MultipleChoiceItem(title="Q1", options=Options(a="A", b="B", c="C", d="D")),
                ]
            )
        }
        
        # Create test image
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image data")
        
        result = analyze_image.invoke({
            "image_paths": str(image_path),
            "question_type": "multiple_choice"
        })
        
        assert "Successfully extracted 1 multiple choice question" in result
        assert "Q1" in result
    
    @patch("src.tools.image_analysis.create_agent")
    @patch("src.tools.image_analysis.ChatOpenAI")
    @patch("src.tools.image_analysis.get_settings")
    def test_analyze_true_false_success(self, mock_get_settings, mock_chat_openai_class, mock_create_agent, tmp_path):
        """Test successful true/false extraction."""
        mock_settings = MagicMock()
        mock_settings.doubao_api_key = "test-key"
        mock_settings.doubao_base_url = "https://test.api"
        mock_settings.doubao_model = "test-model"
        mock_settings.doubao_max_tokens = 4096
        mock_get_settings.return_value = mock_settings
        
        mock_llm = MagicMock()
        mock_chat_openai_class.return_value = mock_llm
        
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "structured_response": TrueFalseResponse(
                questions=[
                    TrueFalseItem(title="Statement 1"),
                    TrueFalseItem(title="Statement 2"),
                ]
            )
        }
        
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image data")
        
        result = analyze_image.invoke({
            "image_paths": str(image_path),
            "question_type": "true_false"
        })
        
        assert "Successfully extracted 2 true false question" in result
        assert "Statement 1" in result
        assert "Statement 2" in result
    
    @patch("src.tools.image_analysis.create_agent")
    @patch("src.tools.image_analysis.ChatOpenAI")
    @patch("src.tools.image_analysis.get_settings")
    def test_analyze_mixed_success(self, mock_get_settings, mock_chat_openai_class, mock_create_agent, tmp_path):
        """Test successful mixed extraction."""
        mock_settings = MagicMock()
        mock_settings.doubao_api_key = "test-key"
        mock_settings.doubao_base_url = "https://test.api"
        mock_settings.doubao_model = "test-model"
        mock_settings.doubao_max_tokens = 4096
        mock_get_settings.return_value = mock_settings
        
        mock_llm = MagicMock()
        mock_chat_openai_class.return_value = mock_llm
        
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "structured_response": MixedResponse(
                multiple_choice_questions=[
                    MultipleChoiceItem(title="MC Question", options=Options(a="A", b="B", c="C", d="D")),
                ],
                true_false_questions=[
                    TrueFalseItem(title="TF Statement 1"),
                    TrueFalseItem(title="TF Statement 2"),
                ]
            )
        }
        
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image data")
        
        result = analyze_image.invoke({
            "image_paths": str(image_path),
            "question_type": "mixed"
        })
        
        assert "Successfully extracted 3 question(s): 1 multiple choice, 2 true/false" in result
        assert "MC Question" in result
        assert "TF Statement 1" in result
        assert "TF Statement 2" in result
        assert "multiple_choice" in result
        assert "true_false" in result
    
    @patch("src.tools.image_analysis.create_agent")
    @patch("src.tools.image_analysis.ChatOpenAI")
    @patch("src.tools.image_analysis.get_settings")
    def test_analyze_mixed_only_multiple_choice(self, mock_get_settings, mock_chat_openai_class, mock_create_agent, tmp_path):
        """Test mixed extraction when only multiple choice found."""
        mock_settings = MagicMock()
        mock_settings.doubao_api_key = "test-key"
        mock_settings.doubao_base_url = "https://test.api"
        mock_settings.doubao_model = "test-model"
        mock_settings.doubao_max_tokens = 4096
        mock_get_settings.return_value = mock_settings
        
        mock_llm = MagicMock()
        mock_chat_openai_class.return_value = mock_llm
        
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "structured_response": MixedResponse(
                multiple_choice_questions=[
                    MultipleChoiceItem(title="Only MC", options=Options(a="A", b="B", c="C", d="D")),
                ],
                true_false_questions=[]
            )
        }
        
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image data")
        
        result = analyze_image.invoke({
            "image_paths": str(image_path),
            "question_type": "mixed"
        })
        
        assert "1 multiple choice, 0 true/false" in result
    
    @patch("src.tools.image_analysis.create_agent")
    @patch("src.tools.image_analysis.ChatOpenAI")
    @patch("src.tools.image_analysis.get_settings")
    def test_analyze_with_multiple_images(self, mock_get_settings, mock_chat_openai_class, mock_create_agent, tmp_path):
        """Test analysis with multiple images."""
        mock_settings = MagicMock()
        mock_settings.doubao_api_key = "test-key"
        mock_settings.doubao_base_url = "https://test.api"
        mock_settings.doubao_model = "test-model"
        mock_settings.doubao_max_tokens = 4096
        mock_get_settings.return_value = mock_settings
        
        mock_llm = MagicMock()
        mock_chat_openai_class.return_value = mock_llm
        
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "structured_response": MultipleChoiceResponse(
                questions=[
                    MultipleChoiceItem(title="Q1", options=Options(a="A", b="B", c="C", d="D")),
                    MultipleChoiceItem(title="Q2", options=Options(a="1", b="2", c="3", d="4")),
                ]
            )
        }
        
        # Create multiple test images
        image1 = tmp_path / "test1.png"
        image2 = tmp_path / "test2.jpg"
        image1.write_bytes(b"fake image 1")
        image2.write_bytes(b"fake image 2")
        
        result = analyze_image.invoke({
            "image_paths": f"{image1},{image2}",
            "question_type": "multiple_choice"
        })
        
        assert "Successfully extracted 2 multiple choice question" in result
        assert "Source images: 2" in result
    
    @patch("src.tools.image_analysis.create_agent")
    @patch("src.tools.image_analysis.ChatOpenAI")
    @patch("src.tools.image_analysis.get_settings")
    def test_analyze_api_error(self, mock_get_settings, mock_chat_openai_class, mock_create_agent, tmp_path):
        """Test handling of API errors."""
        mock_settings = MagicMock()
        mock_settings.doubao_api_key = "test-key"
        mock_settings.doubao_base_url = "https://test.api"
        mock_settings.doubao_model = "test-model"
        mock_settings.doubao_max_tokens = 4096
        mock_get_settings.return_value = mock_settings
        
        mock_llm = MagicMock()
        mock_chat_openai_class.return_value = mock_llm
        
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.side_effect = Exception("API Error")
        
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image data")
        
        result = analyze_image.invoke({
            "image_paths": str(image_path),
            "question_type": "multiple_choice"
        })
        
        assert "Error during image analysis" in result
        assert "API Error" in result
