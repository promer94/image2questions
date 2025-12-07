# Development Plan: LangChain Question Extraction Agent

**Created:** December 7, 2025  
**Status:** In Progress

---

## Overview

This document outlines the development plan for building a LangChain-based agent system that automatically extracts questions from images and outputs them in both JSON and Word document formats.

---

## Current State Assessment

### Existing Components (Working)
- ✅ `image_to_json.py` - Multiple-choice image recognition
- ✅ `image_to_word.py` - Multiple-choice image → Word
- ✅ `image_to_word_truefalse.py` - True/False image → Word
- ✅ `json_to_word.py` - JSON → Word conversion
- ✅ `word_to_json.py` - Word → JSON (multiple-choice)
- ✅ `word_to_json_truefalse.py` - Word → JSON (true/false)

### To Be Built (per DESIGN.md)
- ❌ LangChain Agent orchestration layer
- ❌ Tool wrappers for existing functions
- ❌ CLI with interactive chat mode
- ❌ Validation system
- ❌ Batch processing tool
- ❌ Configuration management
- ❌ Proper project structure (`src/` layout)

---

## Phase 1: Project Restructuring & Foundation

| Task | Description | Status |
|------|-------------|--------|
| 1.1 | Create `src/` directory structure per Appendix A | ✅ |
| 1.2 | Create Pydantic models in `src/models/questions.py` | ✅ |
| 1.3 | Create configuration system in `src/models/config.py` | ✅ |
| 1.4 | Add `.env.example` file | ✅ |
| 1.5 | Create utility modules (`image_utils.py`, `file_utils.py`) | ✅ |
| 1.6 | Update `pyproject.toml` with dependencies | ✅ |
| 1.7 | Create test scaffolding | ✅ |
| 1.8 | Create CLI placeholder | ✅ |

### Deliverables Created
```
src/
├── __init__.py
├── cli.py
├── models/
│   ├── __init__.py
│   ├── questions.py
│   └── config.py
├── tools/
│   └── __init__.py
├── agent/
│   └── __init__.py
└── utils/
    ├── __init__.py
    ├── image_utils.py
    └── file_utils.py

tests/
├── conftest.py
├── test_models.py
├── test_utils.py
└── fixtures/
    └── .gitkeep

.env.example
```

---

## Phase 2: LangChain Tools Implementation ✅

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| 2.1 | Create `ImageAnalysisTool` wrapping existing recognition logic | High | ✅ |
| 2.2 | Create `JSONGeneratorTool` with append mode support | High | ✅ |
| 2.3 | Create `WordGeneratorTool` for both question types | High | ✅ |
| 2.4 | Create `ValidationTool` with confidence scoring | Medium | ✅ |
| 2.5 | Create `BatchProcessingTool` | Medium | ✅ |
| 2.6 | Write unit tests for all tools | High | ✅ |

### Deliverables Created
```
src/tools/
├── __init__.py
├── base.py               # Base tool class and shared utilities
├── image_analysis.py     # Vision model integration
├── json_generator.py     # JSON output tool
├── word_generator.py     # Word document tool
├── validation.py         # Question validation
└── batch_processor.py    # Directory processing

tests/
├── test_tools.py         # Tool unit tests (46 tests, all passing)
```

### Tool Specifications (Implemented)

#### 2.1 ImageAnalysisTool (`analyze_image`)
- **Input:** Image path, question type (auto/multiple_choice/true_false)
- **Output:** List of extracted questions
- **Features:**
  - Auto-detect question type
  - Support for multiple images
  - Confidence scoring
  - Wraps existing `recognize_questions_from_images()` logic

#### 2.2 JSONGeneratorTool
- **Input:** Questions list, output path, append mode flag
- **Output:** File path and statistics
- **Features:**
  - Append or overwrite mode
  - JSON validation
  - Pretty formatting

#### 2.3 WordGeneratorTool
- **Input:** Questions list, output path, template style
- **Output:** File path and statistics
- **Features:**
  - Table layout for multiple choice
  - List layout for true/false
  - Proper numbering and formatting

#### 2.4 ValidationTool
- **Input:** Extracted questions
- **Output:** Validation report with confidence scores
- **Checks:**
  - Non-empty title
  - Complete options (A-D for multiple choice)
  - Reasonable text lengths
  - No duplicate options

#### 2.5 BatchProcessingTool
- **Input:** Directory path, question type, recursive flag
- **Output:** Aggregated results from all images
- **Features:**
  - Progress tracking
  - Error handling per image
  - Summary statistics

---

## Phase 3: Agent Orchestration ✅

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| 3.1 | Create agent prompt templates in `src/agent/prompts.py` | High | ✅ |
| 3.2 | Implement agent configuration in `src/agent/agent.py` | High | ✅ |
| 3.3 | Add conversation memory in `src/agent/agent.py` | Medium | ✅ |
| 3.4 | Integrate all tools with create_agent | High | ✅ |
| 3.5 | Test workflow patterns (single, batch, interactive) | High | ✅ |

### Implementation Notes
- Agent uses **LangChain v1's `create_agent`** with LangGraph under the hood
- Short-term memory implemented via **`InMemorySaver`** checkpointer
- Conversation history persisted via `thread_id` configuration
- Environment variables: `DOUBAO_API_KEY`, `AGENT_MODEL`, `DOUBAO_MODEL`
- System prompts available in Chinese (default) and English

### Architecture
```
LangChain create_agent (high-level API)
         ↓
    LangGraph Runtime
         ↓
  InMemorySaver (checkpointer)
         ↓
  Thread-scoped short-term memory
```

### Deliverables Created
```
src/agent/
├── __init__.py    # Public API exports
├── agent.py       # QuestionExtractionAgent with InMemorySaver for short-term memory
├── prompts.py     # System prompts (Chinese/English), templates
```

### Key Classes and Functions

#### QuestionExtractionAgent
The main agent class with conversation memory support:

```python
from src.agent import QuestionExtractionAgent

# Create agent
agent = QuestionExtractionAgent()

# Multi-turn conversation (memory preserved)
response1 = agent.chat("提取 test.jpg 中的选择题")
response2 = agent.chat("验证这些题目")  # Remembers context
response3 = agent.chat("保存为Word")     # Uses previous results

# Start new conversation (clear memory)
agent.new_conversation()
```

#### Short-Term Memory Implementation
- Uses LangGraph's `InMemorySaver` as checkpointer
- Memory is thread-scoped via `thread_id`
- Each conversation thread maintains its own history
- Memory persists during the session, cleared on restart

### Workflow Patterns Implemented

#### Pattern 1: Single Image Extraction
```
User: "Extract questions from image.jpg"
  ↓
Agent: Analyzes request → Determines output needs
  ↓
Tool: ImageAnalysisTool(image.jpg, auto_detect_type=True)
  ↓
Tool: ValidationTool(extracted_questions)
  ↓
Agent: Reviews validation results
  ↓
Tool: JSONGeneratorTool(questions, "output.json")
  ↓
Tool: WordGeneratorTool(questions, "output.docx")
  ↓
Agent: Reports success with file paths and summary
```

#### Pattern 2: Batch Processing
```
User: "Extract all questions from the multiple-choice-images folder"
  ↓
Agent: Identifies batch operation need
  ↓
Tool: BatchProcessingTool(directory, question_type)
  ↓
Agent: Monitors progress
  ↓
Tool: JSONGeneratorTool + WordGeneratorTool
  ↓
Agent: Reports batch statistics
```

#### Pattern 3: Interactive Refinement
```
User: "Extract questions from test.jpg"
  ↓
Tool: ImageAnalysisTool
  ↓
Agent: "Found 5 questions. Save them?"
  ↓
User: "Yes, but only JSON"
  ↓
Tool: JSONGeneratorTool
```

---

## Phase 4: CLI & User Interface ✅

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| 4.1 | Implement `extract` command fully | High | ✅ |
| 4.2 | Implement `batch` command fully | High | ✅ |
| 4.3 | Implement `interactive` chat mode | High | ✅ |
| 4.4 | Add progress indicators with `rich` | Medium | ✅ |
| 4.5 | Add colorized output and error handling | Medium | ✅ |
| 4.6 | Add memory display and tool call visualization | High | ✅ |

### CLI Features Implemented

#### Interactive Mode Features
- **Conversation Memory Display**: `/memory` command shows full conversation history
- **Tool Call Visualization**: Real-time display of tool calls with arguments
- **Tool Result Display**: Shows tool execution results with success/failure status
- **Streaming Support**: Agent responses streamed in real-time
- **Session Management**: `/clear` to start new conversation, `/exit` to quit

#### Display Helpers
- `display_memory()` - Shows conversation history in a formatted table
- `display_tool_call()` - Displays tool being called with its arguments
- `display_tool_result()` - Shows tool execution results
- `display_agent_response()` - Renders final agent response with Markdown
- `display_config()` - Shows current configuration settings
- `display_tools_list()` - Lists all available agent tools

#### Interactive Commands
| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/memory` | Display conversation memory |
| `/clear` | Start a new conversation |
| `/config` | Show current configuration |
| `/tools` | List available tools |
| `/verbose` | Toggle verbose mode |
| `/debug` | Toggle debug mode |
| `/exit` | Exit interactive mode |

### CLI Commands

```bash
# Extract from single image
question-agent extract image.jpg -t multiple_choice -j output.json -w output.docx

# Extract with verbose output (shows tool calls)
question-agent extract image.jpg -v

# Batch process directory
question-agent batch ./images/ -t auto -j all_questions.json --recursive

# Interactive mode (default: Chinese, streaming, verbose)
question-agent interactive

# Interactive mode with English and no streaming
question-agent interactive --language en --no-stream

# Interactive mode with quiet output (no tool details)
question-agent interactive --quiet

# Show configuration
question-agent config

# List available tools
question-agent tools
```

### Architecture
```
CLI (src/cli.py)
├── Display Helpers
│   ├── display_welcome()      # Welcome message
│   ├── display_memory()       # Memory/history view
│   ├── display_tool_call()    # Tool invocation display
│   ├── display_tool_result()  # Tool result display
│   ├── display_agent_response() # Final response
│   └── display_config()       # Configuration view
│
├── Chat Handlers
│   ├── stream_chat()          # Streaming mode
│   └── invoke_chat()          # Non-streaming mode
│
└── Commands
    ├── extract               # Single/multiple image processing
    ├── batch                 # Directory batch processing
    ├── interactive           # Chat mode
    ├── config                # Show configuration
    └── tools                 # List tools
```

---

## Phase 5: Testing & Quality

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| 5.1 | Create test fixtures (sample images, expected outputs) | High | ⬜ |
| 5.2 | Unit tests for tools (`test_tools.py`) | High | ⬜ |
| 5.3 | Integration tests for agent (`test_agent.py`) | Medium | ⬜ |
| 5.4 | Add pytest configuration | Medium | ⬜ |
| 5.5 | Achieve 80%+ code coverage | Medium | ⬜ |

### Test Structure
```
tests/
├── conftest.py           # Shared fixtures
├── test_models.py        # Model tests
├── test_utils.py         # Utility tests
├── test_tools.py         # Tool tests
├── test_agent.py         # Agent integration tests
├── test_cli.py           # CLI tests
└── fixtures/
    ├── sample_multiple_choice.jpg
    ├── sample_true_false.jpg
    └── expected_outputs/
        ├── sample_multiple_choice.json
        └── sample_true_false.json
```

---

## Phase 6: Documentation & Polish

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| 6.1 | Update `README.md` with full usage guide | High | ⬜ |
| 6.2 | Create `docs/USER_GUIDE.md` | Medium | ⬜ |
| 6.3 | Create `docs/API_REFERENCE.md` | Medium | ⬜ |
| 6.4 | Add `CHANGELOG.md` | Low | ⬜ |
| 6.5 | Add `CONTRIBUTING.md` | Low | ⬜ |

---

## Timeline

```
Week 1: Phase 1 - Project Structure     
Week 2: Phase 2 - LangChain Tools       
Week 3: Phase 3 - Agent Orchestration   
Week 4: Phase 4 - CLI Interface
Week 5: Phase 5 - Testing
Week 6: Phase 6 - Documentation
```

---

## Dependencies

### Production Dependencies
```toml
dependencies = [
    "langchain>=1.1.0",
    "langchain-openai>=1.1.0",
    "langgraph>=0.4.0",
    "langgraph-checkpoint>=2.0.0",
    "openai>=1.0.0",
    "python-docx>=1.2.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "click>=8.1.0",
    "rich>=13.0.0",
]
```

### Development Dependencies
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
]
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Extraction Accuracy | >95% for clear images |
| Processing Speed | <10 seconds per image |
| Test Coverage | 80%+ |
| API Reliability | 99.9% uptime |

---

## Quick Commands

```powershell
# Install dependencies
uv sync

# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run CLI
uv run question-agent --help

# Check configuration
uv run question-agent config
```

---

## Notes

- The existing standalone scripts (`image_to_json.py`, etc.) will remain functional for backward compatibility
- New `src/` modules will provide the modular, agent-based approach
- Environment variables are managed through `.env` file (see `.env.example`)
