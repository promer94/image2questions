# LangChain Question Extraction Agent - Design Document

**Version:** 1.0  
**Date:** December 7, 2025  
**Status:** Design Phase

---

## 1. Executive Summary

This document outlines the design for a LangChain-based agent system that automatically extracts questions from images and outputs them in both JSON and Word document formats. The system supports two question types: multiple-choice questions (with options A-D) and True/False questions.

### 1.1 Goals
- Provide an intelligent, conversational interface for question extraction
- Support multiple-choice and True/False question formats
- Generate structured JSON output for programmatic use
- Create formatted Word documents for human readability
- Enable batch processing of multiple images
- Maintain extensibility for future question types

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  (CLI / API / Interactive Chat)                             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              LangChain Agent Orchestrator                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Agent Executor (ReAct / Function Calling)           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────┬────────────────────────────────────────┬──────────┘
          │                                        │
┌─────────▼──────────────┐            ┌───────────▼───────────┐
│   Tools Registry       │            │   Memory Manager      │
│  ┌──────────────────┐  │            │  - Conversation       │
│  │ Image Analysis   │  │            │  - Extraction History │
│  │ Tool             │  │            └───────────────────────┘
│  └──────────────────┘  │
│  ┌──────────────────┐  │
│  │ JSON Generator   │  │
│  │ Tool             │  │
│  └──────────────────┘  │
│  ┌──────────────────┐  │
│  │ Word Generator   │  │
│  │ Tool             │  │
│  └──────────────────┘  │
│  ┌──────────────────┐  │
│  │ Validation Tool  │  │
│  └──────────────────┘  │
└────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────┐
│               External Services Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Vision LLM   │  │ File System  │  │ Output          │  │
│  │ (Doubao/     │  │              │  │ Formatters      │  │
│  │  GPT-4V)     │  │              │  │ (JSON/Word)     │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

#### 2.2.1 Agent Orchestrator
- **LangChain Agent**: ReAct-style agent with function calling capabilities
- **LLM**: OpenAI-compatible model (Doubao or GPT-4)
- **Prompt Template**: System instructions for question extraction workflow
- **Decision Engine**: Determines tool sequence and handles user intents

#### 2.2.2 Tools Suite

**Tool 1: Image Analysis Tool**
- **Purpose**: Extract questions from images using vision models
- **Input**: Image path or base64 encoded image
- **Output**: Structured question data (Pydantic models)
- **Implementation**: Calls vision LLM with specialized prompts

**Tool 2: JSON Generator Tool**
- **Purpose**: Save extracted questions to JSON format
- **Input**: Question data structures
- **Output**: JSON file path
- **Features**: Append mode, validation, formatting

**Tool 3: Word Document Generator Tool**
- **Purpose**: Create formatted Word documents
- **Input**: Question data structures
- **Output**: Word document path
- **Features**: Custom formatting, numbering, table layouts

**Tool 4: Validation Tool**
- **Purpose**: Validate extracted questions
- **Input**: Raw extraction results
- **Output**: Validation report with confidence scores
- **Features**: Schema validation, completeness checks

**Tool 5: Batch Processing Tool**
- **Purpose**: Process multiple images in one operation
- **Input**: Directory path or list of image paths
- **Output**: Aggregated results
- **Features**: Parallel processing, progress tracking

---

## 3. Data Models

### 3.1 Question Type Schema

#### Multiple Choice Question
```python
from pydantic import BaseModel, Field
from typing import Optional

class MultipleChoiceOptions(BaseModel):
    a: str = Field(description="Option A text")
    b: str = Field(description="Option B text")
    c: str = Field(description="Option C text")
    d: str = Field(description="Option D text")

class MultipleChoiceQuestion(BaseModel):
    title: str = Field(description="The question text")
    options: MultipleChoiceOptions
    correct_answer: Optional[str] = Field(
        None, 
        description="Correct answer key (a/b/c/d)"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Additional metadata (difficulty, category, etc.)"
    )
```

#### True/False Question
```python
class TrueFalseQuestion(BaseModel):
    title: str = Field(description="The statement to evaluate")
    correct_answer: Optional[bool] = Field(
        None,
        description="Correct answer (true/false)"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Additional metadata"
    )
```

#### Unified Question Container
```python
from enum import Enum
from typing import Union

class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"

class Question(BaseModel):
    type: QuestionType
    data: Union[MultipleChoiceQuestion, TrueFalseQuestion]
    source_image: str = Field(description="Original image filename")
    extraction_timestamp: str
    confidence_score: Optional[float] = Field(
        None,
        description="Extraction confidence (0-1)"
    )

class QuestionSet(BaseModel):
    questions: list[Question]
    total_count: int
    metadata: dict = Field(
        default_factory=dict,
        description="Batch-level metadata"
    )
```

---

## 4. Agent Design

### 4.1 Agent Configuration

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Agent System Prompt
AGENT_SYSTEM_PROMPT = """You are an intelligent question extraction assistant. 
Your role is to help users extract questions from images and save them in 
structured formats (JSON and Word documents).

You have access to the following capabilities:
1. Analyze images to extract multiple-choice or true/false questions
2. Save extracted questions to JSON files
3. Generate formatted Word documents
4. Validate extraction results
5. Process multiple images in batch mode

When the user provides an image:
1. Determine the question type (multiple choice or true/false)
2. Extract all questions accurately
3. Validate the extraction quality
4. Save to requested formats (JSON and/or Word)
5. Provide a summary of the extraction results

Always confirm question type and output preferences with the user if unclear.
Be conversational and helpful throughout the process."""

# Tools list
tools = [
    image_analysis_tool,
    json_generator_tool,
    word_generator_tool,
    validation_tool,
    batch_processing_tool
]

# Create agent
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)
```

### 4.2 Workflow Patterns

#### Pattern 1: Single Image Extraction
```
User: "Extract questions from image.jpg"
  ↓
Agent: Analyzes request → Determines output needs
  ↓
Tool: image_analysis_tool(image.jpg, auto_detect_type=True)
  ↓
Tool: validation_tool(extracted_questions)
  ↓
Agent: Reviews validation results
  ↓
Tool: json_generator_tool(questions, "output.json")
  ↓
Tool: word_generator_tool(questions, "output.docx")
  ↓
Agent: Reports success with file paths and summary
```

#### Pattern 2: Batch Processing
```
User: "Extract all questions from the multiple-choice-images folder"
  ↓
Agent: Identifies batch operation need
  ↓
Tool: batch_processing_tool(
    directory="multiple-choice-images/",
    question_type="multiple_choice"
)
  ↓
Agent: Monitors progress
  ↓
Tool: json_generator_tool(all_questions, "batch_results.json")
  ↓
Tool: word_generator_tool(all_questions, "batch_results.docx")
  ↓
Agent: Reports batch statistics
```

#### Pattern 3: Interactive Refinement
```
User: "Extract questions from test.jpg"
  ↓
Tool: image_analysis_tool(test.jpg)
  ↓
Agent: "I found 5 multiple choice questions. Should I save them?"
  ↓
User: "Yes, but only save to JSON"
  ↓
Tool: json_generator_tool(questions, "test.json")
  ↓
Agent: "Saved 5 questions to test.json"
```

---

## 5. Tool Implementation Details

### 5.1 Image Analysis Tool

```python
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class ImageAnalysisInput(BaseModel):
    image_path: str = Field(description="Path to the image file")
    question_type: str = Field(
        default="auto",
        description="Question type: 'multiple_choice', 'true_false', or 'auto'"
    )
    extract_answers: bool = Field(
        default=False,
        description="Whether to extract correct answers if visible"
    )

class ImageAnalysisTool(BaseTool):
    name: str = "analyze_image_for_questions"
    description: str = """Analyzes an image to extract questions. 
    Use this when the user provides an image path and wants to extract 
    questions from it. Returns structured question data."""
    args_schema: Type[BaseModel] = ImageAnalysisInput
    
    def _run(self, image_path: str, question_type: str = "auto", 
             extract_answers: bool = False) -> dict:
        """
        1. Load and encode image
        2. Call vision LLM with appropriate prompt
        3. Parse response into structured format
        4. Return questions with metadata
        """
        # Implementation here
        pass
```

### 5.2 JSON Generator Tool

```python
class JSONGeneratorInput(BaseModel):
    questions: list[dict] = Field(description="List of question objects")
    output_path: str = Field(description="Output JSON file path")
    append_mode: bool = Field(
        default=False,
        description="Append to existing file instead of overwriting"
    )

class JSONGeneratorTool(BaseTool):
    name: str = "save_questions_to_json"
    description: str = """Saves extracted questions to a JSON file.
    Use this after questions have been extracted and validated."""
    args_schema: Type[BaseModel] = JSONGeneratorInput
    
    def _run(self, questions: list[dict], output_path: str, 
             append_mode: bool = False) -> dict:
        """
        1. Load existing data if append_mode=True
        2. Merge or replace question data
        3. Format JSON with proper indentation
        4. Write to file
        5. Return success status and file info
        """
        # Implementation here
        pass
```

### 5.3 Word Generator Tool

```python
class WordGeneratorInput(BaseModel):
    questions: list[dict] = Field(description="List of question objects")
    output_path: str = Field(description="Output Word document path")
    template_style: str = Field(
        default="table_layout",
        description="Document style: 'table_layout' or 'list_format'"
    )

class WordGeneratorTool(BaseTool):
    name: str = "generate_word_document"
    description: str = """Creates a formatted Word document from questions.
    Use this to generate human-readable question documents."""
    args_schema: Type[BaseModel] = WordGeneratorInput
    
    def _run(self, questions: list[dict], output_path: str,
             template_style: str = "table_layout") -> dict:
        """
        1. Create new Word document
        2. Apply formatting based on question type
        3. Add questions with proper numbering
        4. Save document
        5. Return file path and statistics
        """
        # Implementation here
        pass
```

---

## 6. Prompt Engineering

### 6.1 Vision Model Prompts

#### Multiple Choice Extraction Prompt
```python
MULTIPLE_CHOICE_PROMPT = """You are a professional question recognition assistant. 
Carefully analyze the image and extract all multiple-choice questions.

Requirements:
1. Identify all multiple-choice questions in the image
2. Extract the question text (title) accurately
3. Extract all four options (A, B, C, D) completely
4. If an option is missing or unclear, use an empty string
5. Preserve original text formatting and special characters
6. If correct answers are marked, note them

Output Format: Return a JSON object matching this structure:
{
    "questions": [
        {
            "title": "question text here",
            "options": {
                "a": "option A text",
                "b": "option B text", 
                "c": "option C text",
                "d": "option D text"
            }
        }
    ]
}

Be thorough and accurate. Double-check all extracted text."""
```

#### True/False Extraction Prompt
```python
TRUE_FALSE_PROMPT = """You are a professional question recognition assistant.
Carefully analyze the image and extract all True/False statements.

Requirements:
1. Identify all True/False questions in the image
2. Extract the complete statement text accurately
3. Preserve original formatting and punctuation
4. If correct answers are marked, note them

Output Format: Return a JSON object matching this structure:
{
    "questions": [
        {
            "title": "statement text here"
        }
    ]
}

Be thorough and accurate. Extract complete statements."""
```

### 6.2 Agent Reasoning Prompts

#### Task Planning Prompt
```python
TASK_PLANNING_PROMPT = """Given the user's request: "{user_input}"

Analyze the request and create an execution plan:
1. What is the user trying to accomplish?
2. What question type(s) are involved?
3. Is this a single image or batch operation?
4. What output formats are needed?
5. Are there any special requirements?

Plan the sequence of tool calls needed to complete this task."""
```

---

## 7. Error Handling & Validation

### 7.1 Validation Strategy

```python
class QuestionValidator:
    """Validates extracted questions for quality and completeness"""
    
    def validate_multiple_choice(self, question: dict) -> ValidationResult:
        """
        Checks:
        - Title is non-empty
        - All 4 options exist
        - Options are non-empty (or explicitly marked as missing)
        - No duplicate options
        - Reasonable text lengths
        """
        pass
    
    def validate_true_false(self, question: dict) -> ValidationResult:
        """
        Checks:
        - Title is non-empty
        - Statement is complete (ends with punctuation)
        - Reasonable text length
        """
        pass
    
    def calculate_confidence(self, question: dict, 
                           extraction_metadata: dict) -> float:
        """
        Calculates confidence score based on:
        - Vision model confidence
        - Completeness of extracted data
        - Text clarity indicators
        - Format consistency
        """
        pass
```

### 7.2 Error Handling

```python
class ExtractionError(Exception):
    """Base exception for extraction errors"""
    pass

class ImageLoadError(ExtractionError):
    """Raised when image cannot be loaded"""
    pass

class ExtractionQualityError(ExtractionError):
    """Raised when extraction quality is too low"""
    pass

class FormatError(ExtractionError):
    """Raised when output format generation fails"""
    pass

# Agent error recovery strategy
ERROR_RECOVERY_STRATEGIES = {
    ImageLoadError: "retry_with_different_encoding",
    ExtractionQualityError: "request_user_confirmation",
    FormatError: "attempt_alternative_format",
}
```

---

## 8. Configuration Management

### 8.1 Configuration Structure

```python
# config.py
from pydantic import BaseModel
from typing import Optional

class VisionModelConfig(BaseModel):
    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.0

class AgentConfig(BaseModel):
    llm_model: str = "gpt-4-turbo"
    max_iterations: int = 10
    verbose: bool = True
    enable_memory: bool = True

class OutputConfig(BaseModel):
    default_json_path: str = "output.json"
    default_word_path: str = "output.docx"
    auto_save: bool = True
    append_by_default: bool = False

class ApplicationConfig(BaseModel):
    vision_model: VisionModelConfig
    agent: AgentConfig
    output: OutputConfig
    log_level: str = "INFO"
```

### 8.2 Environment Variables

```bash
# .env.example
# Vision Model Configuration
DOUBAO_API_KEY=your_api_key_here
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=doubao-seed-1-6-251015

# Agent LLM Configuration  
OPENAI_API_KEY=your_openai_key_here
AGENT_MODEL=gpt-4-turbo

# Output Configuration
DEFAULT_OUTPUT_DIR=./output
AUTO_SAVE=true

# Logging
LOG_LEVEL=INFO
```

---

## 9. User Interface

### 9.1 CLI Interface

```python
# cli.py
import click
from rich.console import Console

@click.group()
def cli():
    """LangChain Question Extraction Agent"""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--type', '-t', 
              type=click.Choice(['auto', 'multiple_choice', 'true_false']),
              default='auto',
              help='Question type')
@click.option('--output-json', '-j', type=click.Path(), 
              help='Output JSON file path')
@click.option('--output-word', '-w', type=click.Path(),
              help='Output Word file path')
def extract(image_path, type, output_json, output_word):
    """Extract questions from a single image"""
    console = Console()
    console.print(f"[bold green]Extracting questions from:[/bold green] {image_path}")
    # Run agent
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--type', '-t',
              type=click.Choice(['auto', 'multiple_choice', 'true_false']),
              default='auto')
def batch(directory, type):
    """Process all images in a directory"""
    # Run batch processing
    pass

@cli.command()
def interactive():
    """Start interactive chat mode"""
    console = Console()
    console.print("[bold cyan]Question Extraction Agent - Interactive Mode[/bold cyan]")
    console.print("Type 'exit' to quit\n")
    
    while True:
        user_input = console.input("[bold yellow]You:[/bold yellow] ")
        if user_input.lower() in ['exit', 'quit']:
            break
        # Process with agent
        pass
```

### 9.2 Interactive Chat Mode

```
$ python cli.py interactive

Question Extraction Agent - Interactive Mode
Type 'exit' to quit

You: I have some multiple choice questions in image1.jpg

Agent: I'll help you extract multiple choice questions from image1.jpg. 
       Let me analyze the image...
       
       ✓ Found 8 multiple choice questions
       ✓ All questions have complete options (A-D)
       ✓ Extraction confidence: 0.95
       
       Would you like me to save these to JSON and Word formats?

You: Yes, save them as chapter1.json and chapter1.docx

Agent: Saving questions...
       ✓ Saved 8 questions to chapter1.json
       ✓ Generated Word document: chapter1.docx
       
       Summary:
       - 8 multiple choice questions extracted
       - Saved in 2 formats
       - All validations passed

You: exit

Goodbye!
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# tests/test_tools.py
def test_image_analysis_tool_multiple_choice():
    """Test extraction of multiple choice questions"""
    pass

def test_image_analysis_tool_true_false():
    """Test extraction of true/false questions"""
    pass

def test_json_generator_append_mode():
    """Test appending to existing JSON files"""
    pass

def test_word_generator_formatting():
    """Test Word document formatting"""
    pass

def test_validation_tool():
    """Test question validation logic"""
    pass
```

### 10.2 Integration Tests

```python
# tests/test_agent.py
def test_single_image_workflow():
    """Test complete workflow for single image"""
    pass

def test_batch_processing_workflow():
    """Test batch processing of multiple images"""
    pass

def test_error_recovery():
    """Test agent error handling and recovery"""
    pass

def test_interactive_refinement():
    """Test multi-turn conversation flow"""
    pass
```

### 10.3 Test Data

```
tests/
├── fixtures/
│   ├── sample_multiple_choice.jpg
│   ├── sample_true_false.jpg
│   ├── low_quality_image.jpg
│   ├── mixed_questions.jpg
│   └── expected_outputs/
│       ├── sample_multiple_choice.json
│       └── sample_true_false.json
```

---

## 11. Performance Considerations

### 11.1 Optimization Strategies

- **Caching**: Cache vision model responses for identical images
- **Parallel Processing**: Process multiple images concurrently in batch mode
- **Streaming**: Stream agent responses for better UX
- **Rate Limiting**: Respect API rate limits with backoff strategies

### 11.2 Performance Metrics

```python
class PerformanceMetrics:
    extraction_time_per_image: float
    validation_time: float
    file_generation_time: float
    api_call_count: int
    token_usage: dict
    success_rate: float
    average_confidence: float
```

---

## 12. Future Enhancements

### 12.1 Planned Features

1. **Additional Question Types**
   - Fill-in-the-blank
   - Matching questions
   - Short answer questions

2. **Advanced Capabilities**
   - Question deduplication
   - Automatic categorization/tagging
   - Difficulty assessment
   - PDF input support

3. **Collaboration Features**
   - Multi-user support
   - Question review workflow
   - Export to quiz platforms (Quizlet, Kahoot, etc.)

4. **Quality Improvements**
   - Active learning for model improvement
   - User feedback integration
   - OCR fallback for low-quality images

### 12.2 Extensibility Points

- **Custom Tools**: Plugin system for adding new tools
- **Format Exporters**: Additional output format support
- **Vision Models**: Support for multiple vision providers
- **Validation Rules**: Customizable validation logic

---

## 13. Deployment

### 13.1 Deployment Options

**Option 1: Local CLI Tool**
```bash
pip install -e .
export DOUBAO_API_KEY=xxx
export OPENAI_API_KEY=xxx
question-agent extract image.jpg --type multiple_choice
```

**Option 2: REST API Service**
```python
# Use FastAPI to expose agent as REST endpoints
# POST /api/extract
# POST /api/batch
# GET /api/status/{job_id}
```

**Option 3: Web Application**
```
Frontend (React/Vue) → FastAPI Backend → LangChain Agent
```

### 13.2 Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY . .

RUN pip install -e .

CMD ["python", "cli.py", "interactive"]
```

---

## 14. Security Considerations

- **API Key Management**: Use environment variables, never commit keys
- **Input Validation**: Validate all file paths and user inputs
- **Output Sanitization**: Sanitize extracted text before saving
- **Access Control**: Implement authentication for API deployments
- **Rate Limiting**: Prevent abuse of vision model APIs

---

## 15. Monitoring & Logging

```python
# logging_config.py
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("question_agent")

# Log important events
logger.info("Image analysis started")
logger.warning("Low confidence extraction")
logger.error("Failed to save JSON")
```

---

## 16. Documentation Plan

1. **README.md**: Quick start guide
2. **DESIGN.md**: This document (architecture and design)
3. **API_REFERENCE.md**: Tool and function documentation
4. **USER_GUIDE.md**: Step-by-step usage instructions
5. **CONTRIBUTING.md**: Development guidelines
6. **CHANGELOG.md**: Version history

---

## 17. Success Metrics

- **Extraction Accuracy**: >95% for clear images
- **Processing Speed**: <10 seconds per image
- **User Satisfaction**: Measured via feedback
- **API Reliability**: 99.9% uptime for deployed services
- **Code Quality**: 80%+ test coverage

---

## 18. References

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Agents Guide](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [python-docx Documentation](https://python-docx.readthedocs.io/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## Appendix A: File Structure

```
image2test/
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py              # Agent configuration
│   │   ├── prompts.py            # Prompt templates
│   │   └── memory.py             # Conversation memory
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── image_analysis.py     # Image analysis tool
│   │   ├── json_generator.py     # JSON output tool
│   │   ├── word_generator.py     # Word output tool
│   │   ├── validation.py         # Validation tool
│   │   └── batch_processor.py    # Batch processing tool
│   ├── models/
│   │   ├── __init__.py
│   │   ├── questions.py          # Pydantic models
│   │   └── config.py             # Configuration models
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py        # Image processing utilities
│   │   ├── file_utils.py         # File operations
│   │   └── validators.py         # Validation helpers
│   └── cli.py                    # CLI interface
├── tests/
│   ├── test_tools.py
│   ├── test_agent.py
│   └── fixtures/
├── docs/
│   ├── DESIGN.md
│   ├── API_REFERENCE.md
│   └── USER_GUIDE.md
├── pyproject.toml
├── README.md
└── .env.example
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-07 | GitHub Copilot | Initial design document |
