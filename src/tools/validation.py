"""
Validation Tool for LangChain Agent.

This tool validates extracted questions and provides confidence scores
based on completeness and quality checks.
"""

import json
from pathlib import Path
from typing import Any

from langchain.tools import tool


# ==================== Validation Rules ====================

MIN_TITLE_LENGTH = 5
MAX_TITLE_LENGTH = 500
MIN_OPTION_LENGTH = 1
MAX_OPTION_LENGTH = 200


def validate_title(title: str, index: int) -> list[dict]:
    """Validate question title.
    
    Args:
        title: The question title to validate
        index: Question index for error reporting
        
    Returns:
        List of validation issues
    """
    issues = []
    
    if not title:
        issues.append({
            "question_index": index,
            "issue_type": "empty_title",
            "message": "Question title is empty",
            "severity": "error"
        })
    elif len(title.strip()) < MIN_TITLE_LENGTH:
        issues.append({
            "question_index": index,
            "issue_type": "short_title",
            "message": f"Title is too short ({len(title.strip())} chars, minimum {MIN_TITLE_LENGTH})",
            "severity": "warning"
        })
    elif len(title) > MAX_TITLE_LENGTH:
        issues.append({
            "question_index": index,
            "issue_type": "long_title",
            "message": f"Title is very long ({len(title)} chars)",
            "severity": "info"
        })
    
    return issues


def validate_multiple_choice_options(options: dict, index: int) -> list[dict]:
    """Validate multiple choice options.
    
    Args:
        options: Dictionary with keys a, b, c, d
        index: Question index for error reporting
        
    Returns:
        List of validation issues
    """
    issues = []
    required_keys = ["a", "b", "c", "d"]
    
    # Check for missing keys
    missing_keys = [k for k in required_keys if k not in options]
    if missing_keys:
        issues.append({
            "question_index": index,
            "issue_type": "missing_options",
            "message": f"Missing options: {', '.join(missing_keys)}",
            "severity": "error"
        })
        return issues
    
    # Check for empty options
    empty_options = [k.upper() for k in required_keys if not options.get(k, "").strip()]
    if empty_options:
        if len(empty_options) == 4:
            issues.append({
                "question_index": index,
                "issue_type": "all_empty_options",
                "message": "All options are empty",
                "severity": "error"
            })
        else:
            issues.append({
                "question_index": index,
                "issue_type": "empty_options",
                "message": f"Empty options: {', '.join(empty_options)}",
                "severity": "warning"
            })
    
    # Check for duplicate options
    option_values = [options.get(k, "").strip().lower() for k in required_keys]
    non_empty_values = [v for v in option_values if v]
    if len(non_empty_values) != len(set(non_empty_values)):
        issues.append({
            "question_index": index,
            "issue_type": "duplicate_options",
            "message": "Some options have duplicate values",
            "severity": "warning"
        })
    
    # Check option lengths
    for key in required_keys:
        value = options.get(key, "")
        if value and len(value) > MAX_OPTION_LENGTH:
            issues.append({
                "question_index": index,
                "issue_type": "long_option",
                "message": f"Option {key.upper()} is very long ({len(value)} chars)",
                "severity": "info"
            })
    
    return issues


def validate_multiple_choice_question(question: dict, index: int) -> list[dict]:
    """Validate a single multiple choice question.
    
    Args:
        question: Question dictionary with title and options
        index: Question index for error reporting
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Validate title
    title = question.get("title", "")
    issues.extend(validate_title(title, index))
    
    # Validate options
    options = question.get("options", question.get("option", {}))
    if not options:
        issues.append({
            "question_index": index,
            "issue_type": "missing_options_field",
            "message": "Question is missing 'options' field",
            "severity": "error"
        })
    else:
        issues.extend(validate_multiple_choice_options(options, index))
    
    return issues


def validate_true_false_question(question: dict, index: int) -> list[dict]:
    """Validate a single true/false question.
    
    Args:
        question: Question dictionary with title
        index: Question index for error reporting
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Validate title
    title = question.get("title", "")
    issues.extend(validate_title(title, index))
    
    return issues


def calculate_confidence_score(questions: list[dict], issues: list[dict]) -> float:
    """Calculate overall confidence score.
    
    Args:
        questions: List of validated questions
        issues: List of validation issues
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not questions:
        return 0.0
    
    # Count issues by severity
    error_count = sum(1 for i in issues if i["severity"] == "error")
    warning_count = sum(1 for i in issues if i["severity"] == "warning")
    
    # Calculate penalty
    total_questions = len(questions)
    error_penalty = error_count * 0.3
    warning_penalty = warning_count * 0.1
    
    # Calculate score
    score = 1.0 - (error_penalty + warning_penalty) / total_questions
    
    return max(0.0, min(1.0, score))


def validate_questions(
    questions: list[dict],
    question_type: str = "multiple_choice"
) -> dict:
    """Validate a list of questions.
    
    Args:
        questions: List of question dictionaries
        question_type: "multiple_choice" or "true_false"
        
    Returns:
        Validation report dictionary
    """
    issues = []
    
    for i, question in enumerate(questions):
        if question_type == "multiple_choice":
            issues.extend(validate_multiple_choice_question(question, i))
        else:
            issues.extend(validate_true_false_question(question, i))
    
    # Check for overall validity
    has_errors = any(issue["severity"] == "error" for issue in issues)
    
    return {
        "is_valid": not has_errors,
        "total_questions": len(questions),
        "issues_count": len(issues),
        "error_count": sum(1 for i in issues if i["severity"] == "error"),
        "warning_count": sum(1 for i in issues if i["severity"] == "warning"),
        "info_count": sum(1 for i in issues if i["severity"] == "info"),
        "confidence_score": calculate_confidence_score(questions, issues),
        "issues": issues
    }


def collect_failed_question_ids(
    questions: list[dict],
    issues: list[dict],
) -> list[str]:
    """Collect IDs of questions that failed validation.

    A question is considered failed if it has at least one issue with
    severity == "error". If a question has no id, a fallback like "Q1"
    is used.
    """
    error_indices = {issue["question_index"] for issue in issues if issue["severity"] == "error"}
    failed_ids: list[str] = []
    for idx in sorted(error_indices):
        if 0 <= idx < len(questions):
            qid = questions[idx].get("id")
            failed_ids.append(qid if qid else f"Q{idx+1}")
    return failed_ids


def collect_failed_question_ids_mixed(data: dict, issues: list[dict]) -> list[str]:
    """Collect failed IDs for mixed question JSON."""
    mc_questions = data.get("multiple_choice", []) or []
    tf_questions = data.get("true_false", []) or []

    mc_error_indices = {
        issue["question_index"]
        for issue in issues
        if issue.get("severity") == "error" and issue.get("question_type") == "multiple_choice"
    }
    tf_error_indices = {
        issue["question_index"]
        for issue in issues
        if issue.get("severity") == "error" and issue.get("question_type") == "true_false"
    }

    failed_ids: list[str] = []
    for idx in sorted(mc_error_indices):
        if 0 <= idx < len(mc_questions):
            qid = mc_questions[idx].get("id")
            failed_ids.append(qid if qid else f"MC_Q{idx+1}")
    for idx in sorted(tf_error_indices):
        if 0 <= idx < len(tf_questions):
            qid = tf_questions[idx].get("id")
            failed_ids.append(qid if qid else f"TF_Q{idx+1}")

    return failed_ids


def validate_mixed_questions(data: dict) -> dict:
    """Validate mixed questions (both multiple choice and true/false).
    
    Args:
        data: Dictionary with 'multiple_choice' and 'true_false' lists
        
    Returns:
        Combined validation report dictionary
    """
    mc_questions = data.get("multiple_choice", [])
    tf_questions = data.get("true_false", [])
    
    # Validate each type separately
    mc_issues = []
    for i, question in enumerate(mc_questions):
        mc_issues.extend(validate_multiple_choice_question(question, i))
    
    tf_issues = []
    for i, question in enumerate(tf_questions):
        tf_issues.extend(validate_true_false_question(question, i))
    
    # Mark issues with question type for clarity
    for issue in mc_issues:
        issue["question_type"] = "multiple_choice"
    for issue in tf_issues:
        issue["question_type"] = "true_false"
    
    all_issues = mc_issues + tf_issues
    all_questions = mc_questions + tf_questions
    
    # Check for overall validity
    has_errors = any(issue["severity"] == "error" for issue in all_issues)
    
    return {
        "is_valid": not has_errors,
        "total_questions": len(all_questions),
        "multiple_choice_count": len(mc_questions),
        "true_false_count": len(tf_questions),
        "issues_count": len(all_issues),
        "error_count": sum(1 for i in all_issues if i["severity"] == "error"),
        "warning_count": sum(1 for i in all_issues if i["severity"] == "warning"),
        "info_count": sum(1 for i in all_issues if i["severity"] == "info"),
        "confidence_score": calculate_confidence_score(all_questions, all_issues),
        "multiple_choice_issues": mc_issues,
        "true_false_issues": tf_issues,
        "issues": all_issues
    }


# ==================== LangChain Tool ====================

@tool
def validate_questions_tool(
    questions_file: str,
    question_type: str = "multiple_choice"
) -> str:
    """Validate extracted questions for completeness and quality.
    
    This tool checks questions for common issues like empty titles,
    missing options, duplicate values, and other quality problems.
    It provides a confidence score and detailed issue reports.
    
    Args:
        questions_file: Path to a JSON file containing questions in one of these formats:
                       - For multiple choice: [{"title": "...", "options": {"a": "...", "b": "...", "c": "...", "d": "..."}}]
                       - For true/false: [{"title": "..."}]
                       - For mixed: {"multiple_choice": [...], "true_false": [...]}
        question_type: Type of questions to validate. Must be one of:
                      - "multiple_choice": Validates title and all four options
                      - "true_false": Validates title only
                      - "mixed": Validates both multiple choice and true/false questions
    
    Returns:
        A validation report including:
        - Whether all questions are valid
        - Confidence score (0.0 to 1.0)
        - Count of errors, warnings, and info messages
        - Detailed list of issues found
    """
    # Read and parse input questions from file
    file_path = Path(questions_file)
    if not file_path.exists():
        return f"Error: File not found: {questions_file}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file: {str(e)}"
    except Exception as e:
        return f"Error: Failed to read file: {str(e)}"

    original_data: Any = data
    
    # Validate question type
    if question_type not in ("multiple_choice", "true_false", "mixed"):
        return f"Error: Invalid question_type '{question_type}'. Must be 'multiple_choice', 'true_false', or 'mixed'."
    
    # Handle mixed type
    if question_type == "mixed":
        if not isinstance(data, dict):
            return "Error: For mixed type, questions must be a JSON object with 'multiple_choice' and 'true_false' arrays"
        
        if "multiple_choice" not in data and "true_false" not in data:
            return "Error: Mixed format requires at least one of 'multiple_choice' or 'true_false' arrays"
        
        report = validate_mixed_questions(data)
        
        # Build result message for mixed type
        lines = [
            f"Validation Report for Mixed Questions",
            f"  - {report['multiple_choice_count']} multiple choice question(s)",
            f"  - {report['true_false_count']} true/false question(s)",
            "=" * 60,
            f"Status: {'✓ VALID' if report['is_valid'] else '✗ INVALID'}",
            f"Confidence Score: {report['confidence_score']:.2f}",
            "",
            f"Issues Summary:",
            f"  - Errors: {report['error_count']}",
            f"  - Warnings: {report['warning_count']}",
            f"  - Info: {report['info_count']}",
        ]
        
        if report["issues"]:
            lines.append("")
            lines.append("Issue Details:")
            
            # Show multiple choice issues
            if report["multiple_choice_issues"]:
                lines.append("  Multiple Choice:")
                for issue in report["multiple_choice_issues"]:
                    severity_icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue["severity"], "•")
                    lines.append(f"    {severity_icon} Q{issue['question_index']+1}: {issue['message']} [{issue['issue_type']}]")
            
            # Show true/false issues
            if report["true_false_issues"]:
                lines.append("  True/False:")
                for issue in report["true_false_issues"]:
                    severity_icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue["severity"], "•")
                    lines.append(f"    {severity_icon} Q{issue['question_index']+1}: {issue['message']} [{issue['issue_type']}]")
        else:
            lines.append("")
            lines.append("No issues found. All questions passed validation.")
        
        failed_ids = collect_failed_question_ids_mixed(original_data, report["issues"])
        lines.append("")
        if failed_ids:
            lines.append(f"Failed Question IDs: {', '.join(failed_ids)}")
        else:
            lines.append("Failed Question IDs: (none)")

        if isinstance(original_data, dict):
            original_data["failed_question_ids"] = failed_ids
            try:
                file_path.write_text(
                    json.dumps(original_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                lines.append("")
                lines.append("Warning: Failed to write failed_question_ids back to JSON.")

        return "\n".join(lines)
    
    # Handle single type (multiple_choice or true_false)
    if not isinstance(data, list):
        # Check if it's the unified structure
        if isinstance(data, dict) and ("multiple_choice" in data or "true_false" in data):
             if question_type == "multiple_choice" and "multiple_choice" in data:
                 data = data["multiple_choice"]
             elif question_type == "true_false" and "true_false" in data:
                 data = data["true_false"]
             else:
                 return f"Error: Input data does not contain '{question_type}' questions"
        else:
            return "Error: Questions must be a JSON array or unified format object"
    
    if not data:
        return "Error: No questions provided. The questions array is empty."
    
    # Run validation
    report = validate_questions(data, question_type)
    
    # Build result message
    lines = [
        f"Validation Report for {report['total_questions']} {question_type.replace('_', ' ')} question(s)",
        "=" * 60,
        f"Status: {'✓ VALID' if report['is_valid'] else '✗ INVALID'}",
        f"Confidence Score: {report['confidence_score']:.2f}",
        "",
        f"Issues Summary:",
        f"  - Errors: {report['error_count']}",
        f"  - Warnings: {report['warning_count']}",
        f"  - Info: {report['info_count']}",
    ]
    
    if report["issues"]:
        lines.append("")
        lines.append("Issue Details:")
        for issue in report["issues"]:
            severity_icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue["severity"], "•")
            lines.append(f"  {severity_icon} Q{issue['question_index']+1}: {issue['message']} [{issue['issue_type']}]")
    else:
        lines.append("")
        lines.append("No issues found. All questions passed validation.")
    
    failed_ids = collect_failed_question_ids(data, report["issues"])
    lines.append("")
    if failed_ids:
        lines.append(f"Failed Question IDs: {', '.join(failed_ids)}")
    else:
        lines.append("Failed Question IDs: (none)")

    if isinstance(original_data, dict):
        original_data["failed_question_ids"] = failed_ids
        try:
            file_path.write_text(
                json.dumps(original_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            lines.append("")
            lines.append("Warning: Failed to write failed_question_ids back to JSON.")
    elif isinstance(original_data, list):
        lines.append("")
        lines.append("Note: Input JSON is an array; failed_question_ids not persisted to avoid changing schema.")

    return "\n".join(lines)


# Export for convenient access
__all__ = [
    "validate_questions_tool",
    "validate_questions",
    "validate_mixed_questions",
]
