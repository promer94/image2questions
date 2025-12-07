# Question Extraction Agent

A LangChain-based agent for extracting questions from images into JSON and Word formats.

## Features

- Extract multiple-choice and true/false questions from images
- Save to JSON and Word formats
- Interactive chat interface
- Batch processing
- Validation of extracted questions

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Configure environment:
   Copy `.env.example` to `.env` and set your API keys.
   ```bash
   cp .env.example .env
   ```

## Usage

### CLI

```bash
# Interactive mode
question-agent interactive

# Extract from image
question-agent extract image.jpg

# Batch process
question-agent batch ./images/
```

### LangSmith Studio

This project is configured for LangSmith Studio, a visual interface for developing and testing LangChain agents.

1. Ensure you have `langgraph-cli` installed (it is included in dev dependencies):
   ```bash
   uv add --dev langgraph-cli
   ```

2. Configure LangSmith credentials in `.env`:
   ```dotenv
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_api_key
   LANGCHAIN_PROJECT=image2test
   ```

3. Start LangSmith Studio:
   ```bash
   uv run langgraph dev
   ```
   This will open the studio in your browser.

## Project Structure

- `src/agent/`: Agent logic and prompts
- `src/tools/`: Tools for image analysis and file generation
- `src/models/`: Pydantic models and configuration
- `src/cli.py`: Command-line interface