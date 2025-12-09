"""
Image Analysis Tool for LangChain Agent.

This tool extracts questions from images using vision models.
Supports both multiple-choice and true/false question types.
Automatically saves extracted questions to JSON file.
"""

import base64
import json
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .json_generator import load_existing_questions
from ..models.config import get_settings


# ==================== Pydantic Models for LLM Response ====================

class Options(BaseModel):
    """Multiple choice options."""
    a: str = ""
    b: str = ""
    c: str = ""
    d: str = ""


class MultipleChoiceItem(BaseModel):
    """A multiple choice question item."""
    title: str
    options: Options


class MultipleChoiceResponse(BaseModel):
    """Response format for multiple choice questions."""
    questions: list[MultipleChoiceItem]


class TrueFalseItem(BaseModel):
    """A true/false question item."""
    title: str


class TrueFalseResponse(BaseModel):
    """Response format for true/false questions."""
    questions: list[TrueFalseItem]


class MixedResponse(BaseModel):
    """Response format for mixed question types (both multiple choice and true/false)."""
    multiple_choice_questions: list[MultipleChoiceItem] = Field(default_factory=list)
    true_false_questions: list[TrueFalseItem] = Field(default_factory=list)


# ==================== System Prompts ====================

MULTIPLE_CHOICE_PROMPT = """你是一个专业的题目识别助手。请仔细分析图片中的选择题，识别出所有题目及其选项。

要求：
1. 识别图片中所有的选择题
2. 提取每道题的题目内容（title）和四个选项（a、b、c、d）
3. 如果某个选项不存在，对应的值设为空字符串
4. 请按照图片中题目出现的顺序提取
5. 不要提取题目的序号，只提取题目内容本身
"""

TRUE_FALSE_PROMPT = """你是一个专业的题目识别助手。请仔细分析图片中的判断题，识别出所有题目。

要求：
1. 识别图片中所有的判断题
2. 提取每道判断题的题目内容（title）
3. 判断题通常是一个陈述句，需要判断其正确或错误
4. 请按照图片中题目出现的顺序提取
5. 不要提取题目的序号，只提取题目内容本身
"""

MIXED_PROMPT = """你是一个专业的题目识别助手。请仔细分析图片中的所有题目，包括选择题和判断题。

要求：
1. 识别图片中所有的选择题和判断题
2. 对于选择题：提取题目内容（title）和四个选项（a、b、c、d），如果某个选项不存在，对应的值设为空字符串
3. 对于判断题：只提取题目内容（title），判断题通常是一个陈述句，需要判断其正确或错误
4. 请按照图片中题目出现的顺序提取
5. 不要提取题目的序号，只提取题目内容本身
6. 正确区分选择题和判断题：选择题有A、B、C、D等选项，判断题没有选项
"""


# ==================== Helper Functions ====================

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type based on file extension."""
    suffix = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(suffix, "image/jpeg")


def validate_image_paths(image_paths: list[str]) -> tuple[list[str], list[str]]:
    """Validate image paths and return valid paths and errors."""
    valid_paths = []
    errors = []
    
    for path in image_paths:
        if not os.path.exists(path):
            errors.append(f"Image not found: {path}")
        elif not os.path.isfile(path):
            errors.append(f"Not a file: {path}")
        else:
            valid_paths.append(path)
    
    return valid_paths, errors


def save_questions_to_json(
    questions: dict,
    output_path: Path,
    append: bool = False,
    pretty: bool = True
) -> tuple[bool, str]:
    """Save questions to a JSON file.
    
    Args:
        questions: Question dictionary
        output_path: Path to save the JSON file
        append: If True, append to existing file
        pretty: Whether to format with indentation
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle append mode
        if append and output_path.exists():
            existing, error = load_existing_questions(output_path)
            if error:
                return False, f"Error loading existing file: {error}"
            
            # Merge existing and new questions
            questions = {
                "multiple_choice": existing.get("multiple_choice", []) + questions.get("multiple_choice", []),
                "true_false": existing.get("true_false", []) + questions.get("true_false", [])
            }
            
            # Merge processed images
            existing_images = existing.get("processed_images", [])
            new_images = questions.get("processed_images", [])
            if existing_images or new_images:
                all_images = existing_images + [img for img in new_images if img not in existing_images]
                questions["processed_images"] = all_images
            
            # Preserve type if present
            if "type" in existing:
                questions["type"] = existing["type"]
        
        # Format JSON
        if pretty:
            content = json.dumps(questions, ensure_ascii=False, indent=2)
        else:
            content = json.dumps(questions, ensure_ascii=False)
        
        # Write file
        output_path.write_text(content, encoding="utf-8")
        
        # Count questions for message
        count = len(questions.get("multiple_choice", [])) + len(questions.get("true_false", []))
        
        return True, f"Saved {count} questions to {output_path}"
        
    except Exception as e:
        return False, f"Error saving file: {str(e)}"


def build_image_content(image_paths: list[str]) -> list[dict]:
    """Build content list with images for the API call."""
    content = []
    
    for image_path in image_paths:
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        content.append({
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{base64_image}",
        })
    
    return content


def extract_multiple_choice(llm: ChatOpenAI, image_paths: list[str]) -> dict:
    """Extract multiple choice questions from images using LangChain agent."""
    content = [{"type": "text", "text": "请识别以下图片中的所有选择题。"}]
    content.extend(build_image_content(image_paths))
    
    # Create agent with ProviderStrategy for native structured output
    agent = create_agent(
        model=llm,
        system_prompt=MULTIPLE_CHOICE_PROMPT,
        tools=[],
        response_format=ProviderStrategy(MultipleChoiceResponse),
    )
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": content}]
    })
    
    result = response["structured_response"]
    
    return {
        "type": "multiple_choice",
        "multiple_choice": [
            {
                "title": q.title,
                "options": {
                    "a": q.options.a,
                    "b": q.options.b,
                    "c": q.options.c,
                    "d": q.options.d,
                },
                "source_image": image_paths,
            }
            for q in result.questions
        ]
    }


def extract_true_false(llm: ChatOpenAI, image_paths: list[str]) -> dict:
    """Extract true/false questions from images using LangChain agent."""
    content = [{"type": "text", "text": "请识别以下图片中的所有判断题。"}]
    content.extend(build_image_content(image_paths))
    
    # Create agent with ProviderStrategy for native structured output
    agent = create_agent(
        model=llm,
        system_prompt=TRUE_FALSE_PROMPT,
        tools=[],
        response_format=ProviderStrategy(TrueFalseResponse),
    )
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": content}]
    })
    
    result = response["structured_response"]
    
    return {
        "type": "true_false",
        "true_false": [{"title": q.title, "source_image": image_paths} for q in result.questions]
    }


def extract_mixed(llm: ChatOpenAI, image_paths: list[str]) -> dict:
    """Extract both multiple choice and true/false questions from images using LangChain agent."""
    content = [{"type": "input_text", "text": "请识别以下图片中的所有题目，包括选择题和判断题。"}]
    content.extend(build_image_content(image_paths))
    
    # Create agent with ProviderStrategy for native structured output
    agent = create_agent(
        model=llm,
        system_prompt=MIXED_PROMPT,
        tools=[],
        response_format=ProviderStrategy(MixedResponse),
    )
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": content}]
    })
    
    result = response["structured_response"]
    
    multiple_choice = [
        {
            "title": q.title,
            "options": {
                "a": q.options.a,
                "b": q.options.b,
                "c": q.options.c,
                "d": q.options.d,
            },
            "source_image": image_paths,
        }
        for q in result.multiple_choice_questions
    ]
    
    true_false = [{"title": q.title, "source_image": image_paths} for q in result.true_false_questions]
    
    return {
        "type": "mixed",
        "multiple_choice": multiple_choice,
        "true_false": true_false
    }


# ==================== LangChain Tool ====================

@tool
def analyze_image(
    image_paths: str,
    output_path: str,
    question_type: str = "multiple_choice",
    append: bool = False
) -> str:
    """Extract questions from images using vision AI and save to JSON file.
    
    This tool analyzes images containing exam questions, extracts them
    into structured data, and saves directly to a JSON file. It supports 
    both multiple-choice questions (with options A-D) and true/false questions.
    
    Args:
        image_paths: Comma-separated list of image file paths to analyze.
                    Example: "path/to/image1.png,path/to/image2.jpg"
        output_path: File path where the JSON will be saved.
                    Will create parent directories if needed.
        question_type: Type of questions to extract. Must be one of:
                      - "multiple_choice": Questions with options A, B, C, D
                      - "true_false": Statement questions for true/false judgment
                      - "mixed": Both multiple choice and true/false questions
        append: If True, append questions to existing file.
               If False, overwrite existing file. Default: False
    
    Returns:
        A string containing the extraction result with:
        - Number of questions found
        - The file path where questions were saved
        - Any errors encountered
    """
    # Parse image paths
    paths = [p.strip() for p in image_paths.split(",") if p.strip()]
    
    if not paths:
        return "Error: No image paths provided. Please provide at least one image path."
    
    # Validate paths
    valid_paths, errors = validate_image_paths(paths)
    
    if not valid_paths:
        return f"Error: No valid images found.\nErrors:\n" + "\n".join(errors)
    
    # Validate question type
    if question_type not in ("multiple_choice", "true_false", "mixed"):
        return f"Error: Invalid question_type '{question_type}'. Must be 'multiple_choice', 'true_false', or 'mixed'."
    
    # Prepare output path
    file_path = Path(output_path)
    if file_path.suffix.lower() != ".json":
        file_path = file_path.with_suffix(".json")
    
    try:
        # Get settings
        settings = get_settings()
        
        # Create LangChain ChatOpenAI client with custom base URL
        llm = ChatOpenAI(
            api_key=settings.doubao_api_key,
            base_url=settings.doubao_base_url,
            model=settings.doubao_model,
            temperature=0.1,
            max_tokens=settings.doubao_max_tokens,
            use_responses_api=True,
        )
        
        # Extract questions based on type
        if question_type == "multiple_choice":
            result_data = extract_multiple_choice(llm, valid_paths)
            total_count = len(result_data["multiple_choice"])
        elif question_type == "true_false":
            result_data = extract_true_false(llm, valid_paths)
            total_count = len(result_data["true_false"])
        else:  # mixed
            result_data = extract_mixed(llm, valid_paths)
            mc_count = len(result_data["multiple_choice"])
            tf_count = len(result_data["true_false"])
            total_count = mc_count + tf_count
        
        # Add processed images to result data
        result_data["processed_images"] = valid_paths
        
        # Save to JSON file
        success, save_message = save_questions_to_json(result_data, file_path, append=append)
        
        if not success:
            return f"Extraction succeeded but failed to save: {save_message}"
        
        # Build result message
        if question_type == "mixed":
            result_lines = [
                f"Successfully extracted {total_count} question(s): {mc_count} multiple choice, {tf_count} true/false.",
                f"Source images: {len(valid_paths)}",
                save_message,
            ]
        else:
            result_lines = [
                f"Successfully extracted {total_count} {question_type.replace('_', ' ')} question(s).",
                f"Source images: {len(valid_paths)}",
                save_message,
            ]
        
        if errors:
            result_lines.append(f"Warnings: {len(errors)} image(s) could not be processed")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"Error during image analysis: {str(e)}"


# Export for convenient access
__all__ = [
    "analyze_image",
    "extract_multiple_choice",
    "extract_true_false",
    "extract_mixed",
    "validate_image_paths",
]
