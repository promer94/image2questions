"""
Image Analysis Tool for LangChain Agent.

This tool extracts questions from images using vision models.
Supports both multiple-choice and true/false question types.
"""

import base64
import os
from pathlib import Path
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..models.config import get_settings
from ..models.questions import QuestionType


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
                }
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
        "true_false": [{"title": q.title} for q in result.questions]
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
            }
        }
        for q in result.multiple_choice_questions
    ]
    
    true_false = [{"title": q.title} for q in result.true_false_questions]
    
    return {
        "type": "mixed",
        "multiple_choice": multiple_choice,
        "true_false": true_false
    }


# ==================== LangChain Tool ====================

@tool
def analyze_image(
    image_paths: str,
    question_type: str = "multiple_choice"
) -> str:
    """Extract questions from images using vision AI.
    
    This tool analyzes images containing exam questions and extracts them
    into structured data. It supports both multiple-choice questions
    (with options A-D) and true/false questions.
    
    Args:
        image_paths: Comma-separated list of image file paths to analyze.
                    Example: "path/to/image1.png,path/to/image2.jpg"
        question_type: Type of questions to extract. Must be one of:
                      - "multiple_choice": Questions with options A, B, C, D
                      - "true_false": Statement questions for true/false judgment
                      - "mixed": Both multiple choice and true/false questions
    
    Returns:
        A string containing the extraction result with:
        - Number of questions found
        - The extracted questions in JSON format
        - Any errors encountered
    """
    import json
    
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
        
        # Build result message
        if question_type == "mixed":
            result_lines = [
                f"Successfully extracted {total_count} question(s): {mc_count} multiple choice, {tf_count} true/false.",
                f"Source images: {len(valid_paths)}",
            ]
        else:
            result_lines = [
                f"Successfully extracted {total_count} {question_type.replace('_', ' ')} question(s).",
                f"Source images: {len(valid_paths)}",
            ]
        
        if errors:
            result_lines.append(f"Warnings: {len(errors)} image(s) could not be processed")
        
        result_lines.append(f"\nExtracted questions:\n{json.dumps(result_data, ensure_ascii=False, indent=2)}")
        
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
