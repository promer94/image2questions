"""
从 Word 文档中提取判断题并保存为 JSON 格式

将 boolean-questions.docx 中的判断题解析出来，整理成 JSON 数组
"""

import argparse
import json
import re
from pathlib import Path

from docx import Document


def extract_truefalse_from_docx(docx_path: Path) -> list[dict]:
    """
    从 Word 文档中提取判断题
    
    Args:
        docx_path: Word 文档路径
        
    Returns:
        判断题列表，每题包含 title
    """
    document = Document(docx_path)
    questions = []
    
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        
        # 跳过空段落
        if not text:
            continue
        
        # 尝试提取判断题内容
        # 判断题格式: 题目内容（    ）
        # 移除末尾的答案括号
        title = re.sub(r"[（(]\s*[）)]\s*$", "", text).strip()
        
        # 移除开头的序号（如 1. 2. 3、等）
        title = re.sub(r"^\s*\d+[\.)、．\s]+", "", title).strip()
        
        # 跳过太短的内容（可能是标题或空内容）
        if len(title) < 5:
            continue
        
        questions.append({
            "title": title
        })
    
    return questions


def main():
    parser = argparse.ArgumentParser(
        description="从 Word 文档中提取判断题并保存为 JSON"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="输入的 Word 文档路径"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出 JSON 文件路径（默认为输入文件名.json）"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="格式化 JSON 输出（默认启用）"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.input.exists():
        print(f"错误: 文件不存在: {args.input}")
        return 1
    
    # 确定输出路径
    output_path = args.output
    if output_path is None:
        output_path = args.input.with_suffix(".json")
    
    try:
        # 提取判断题
        print(f"正在从 {args.input} 提取判断题...")
        questions = extract_truefalse_from_docx(args.input)
        
        if not questions:
            print("未提取到任何判断题")
            return 1
        
        print(f"成功提取 {len(questions)} 道判断题")
        
        # 保存为 JSON
        indent = 2 if args.pretty else None
        json_output = json.dumps(questions, ensure_ascii=False, indent=indent)
        output_path.write_text(json_output, encoding="utf-8")
        
        print(f"JSON 已保存到: {output_path}")
        
        # 打印预览
        print("\n提取结果预览:")
        for i, q in enumerate(questions[:5], 1):
            title_preview = q['title'][:50] + "..." if len(q['title']) > 50 else q['title']
            print(f"  {i}. {title_preview}")
        
        if len(questions) > 5:
            print(f"\n  ... 共 {len(questions)} 道题目")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
