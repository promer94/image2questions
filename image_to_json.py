"""
使用豆包模型识别图片中的选择题，并输出为 JSON 格式
"""

import argparse
import base64
import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel


class Options(BaseModel):
    a: str
    b: str
    c: str
    d: str


class QuestionItem(BaseModel):
    title: str
    options: Options


class QuestionList(BaseModel):
    questions: list[QuestionItem]


# 豆包模型配置
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "a871c6fc-c014-4451-9db5-125400419b25")
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
# 豆包视觉模型 endpoint，需要替换为你的 endpoint id
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-seed-1-6-251015")

SYSTEM_PROMPT = """你是一个专业的题目识别助手。请仔细分析图片中的选择题，识别出所有题目及其选项。

要求：
1. 识别图片中所有的选择题
2. 提取每道题的题目内容（title）和四个选项（a、b、c、d）
3. 如果某个选项不存在，对应的值设为空字符串
"""


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
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


def recognize_questions_from_images(image_paths: list[str]) -> list[dict]:
    """
    使用豆包模型识别图片中的选择题
    
    Args:
        image_paths: 图片文件路径列表
        
    Returns:
        识别出的选择题列表
    """
    if not DOUBAO_API_KEY:
        raise ValueError("请设置环境变量 DOUBAO_API_KEY")
    
    client = OpenAI(
        api_key=DOUBAO_API_KEY,
        base_url=DOUBAO_BASE_URL,
    )
    
    # 构建包含多张图片的消息内容
    content = []
    
    # 添加文字提示
    content.append({
        "type": "text",
        "text": "请识别以下图片中的所有选择题。"
    })
    
    # 添加所有图片
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"警告: 图片文件不存在: {image_path}")
            continue
            
        base64_image = encode_image_to_base64(image_path)
        mime_type = get_image_mime_type(image_path)
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })
    
    if len(content) == 1:
        raise ValueError("没有有效的图片文件")
    
    # 使用 Chat Completions API 的结构化输出
    response = client.beta.chat.completions.parse(
        model=DOUBAO_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        temperature=0.1,  # 使用较低的温度以获得更稳定的输出
        response_format=QuestionList,
    )
    
    # 解析响应
    result = response.choices[0].message.parsed
    
    # 转换为字典列表格式
    questions = [
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
    
    return questions


def main():
    parser = argparse.ArgumentParser(
        description="使用豆包模型识别图片中的选择题"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="要识别的图片文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 JSON 文件路径（不指定则输出到控制台）"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="格式化输出 JSON"
    )
    
    args = parser.parse_args()
    
    try:
        questions = recognize_questions_from_images(args.images)
        
        # 格式化输出
        if args.pretty:
            output = json.dumps(questions, ensure_ascii=False, indent=2)
        else:
            output = json.dumps(questions, ensure_ascii=False)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"结果已保存到: {args.output}")
        else:
            print(output)
            
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
