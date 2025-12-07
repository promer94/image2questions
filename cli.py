#!/usr/bin/env python
"""Simple CLI for extracting questions from images using the agent."""

import argparse
import sys
import json
from pathlib import Path

from src.agent.agent import extract_questions


def main():
    parser = argparse.ArgumentParser(
        description="Extract questions from images using AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py image1.jpg image2.png
  python cli.py exam.jpg --type multiple_choice
  python cli.py test.png --output-json questions.json
  python cli.py test.png --output-word questions.docx
  python cli.py *.jpg --type true_false --output-json output.json --output-word output.docx
        """,
    )

    parser.add_argument(
        "images",
        nargs="+",
        help="Image file paths to extract questions from",
    )

    parser.add_argument(
        "-t", "--type",
        choices=["auto", "multiple_choice", "true_false", "mixed"],
        default="auto",
        help="Question type to extract (default: auto)",
    )

    parser.add_argument(
        "-j", "--output-json",
        metavar="FILE",
        help="Save extracted questions to JSON file",
    )

    parser.add_argument(
        "-w", "--output-word",
        metavar="FILE",
        help="Save extracted questions to Word document",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate image paths
    image_paths = []
    for img_path in args.images:
        path = Path(img_path)
        if not path.exists():
            print(f"Error: Image file not found: {img_path}", file=sys.stderr)
            sys.exit(1)
        if not path.is_file():
            print(f"Error: Not a file: {img_path}", file=sys.stderr)
            sys.exit(1)
        image_paths.append(str(path.absolute()))

    if args.verbose:
        print(f"Processing {len(image_paths)} image(s)...")
        print(f"Question type: {args.type}")
        if args.output_json:
            print(f"JSON output: {args.output_json}")
        if args.output_word:
            print(f"Word output: {args.output_word}")
        print()

    try:
        result = extract_questions(
            image_paths=image_paths,
            question_type=args.type,
            output_json=args.output_json,
            output_word=args.output_word,
        )

        print("Response:")
        print(result["response"])
        print()

        if result.get("questions"):
            print(f"\nExtracted {len(result['questions'])} question(s)")
            if args.verbose:
                print("\nQuestions (JSON):")
                print(json.dumps(result["questions"], ensure_ascii=False, indent=2))

    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print("\nMake sure you have set the required environment variables:", file=sys.stderr)
        print("  - DOUBAO_API_KEY: API key for the Doubao model", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
