"""
Command-line interface for the Question Extraction Agent.

This module provides CLI commands for extracting questions from images
and generating output in JSON and Word formats.

Features:
- Interactive chat mode with memory display
- Streaming output of agent responses
- Tool call and result visualization
- Conversation history display
"""

import os
import sys
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()


# ============================================================================
# Display Helpers
# ============================================================================

def display_welcome():
    """Display welcome message and help info."""
    welcome_text = """
# 🤖 Question Extraction Agent

Welcome to the interactive chat mode! You can:

- Extract questions from images: `从 image.jpg 中提取题目`
- Batch process: `处理 images/ 目录中的所有图片`
- Save to files: `保存为 output.json 和 output.docx`
- Validate questions: `验证提取的题目质量`

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show this help message |
| `/memory` | Display conversation memory |
| `/clear` | Start a new conversation |
| `/config` | Show current configuration |
| `/tools` | List available tools |
| `/exit` or `/quit` | Exit interactive mode |

Type your message and press Enter to chat with the agent.
"""
    console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))


def display_memory(agent, thread_id: str):
    """Display current conversation memory/history."""
    try:
        history = agent.get_conversation_history(thread_id)
        
        if not history:
            console.print("[yellow]No conversation history yet.[/yellow]")
            return
        
        console.print()
        console.print(Panel("[bold]📝 Conversation Memory[/bold]", border_style="cyan"))
        
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            expand=True,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Role", width=10)
        table.add_column("Content", overflow="fold")
        table.add_column("Tool Info", width=25, overflow="fold")
        
        for i, msg in enumerate(history, 1):
            role = _get_message_role(msg)
            content = _get_message_content(msg)
            tool_info = _get_tool_info(msg)
            
            # Color code by role
            role_style = {
                "user": "green",
                "assistant": "blue",
                "tool": "yellow",
                "system": "magenta",
                "human": "green",
                "ai": "blue",
            }.get(role, "white")
            
            table.add_row(
                str(i),
                f"[{role_style}]{role}[/{role_style}]",
                content[:200] + "..." if len(content) > 200 else content,
                tool_info,
            )
        
        console.print(table)
        console.print(f"\n[dim]Total messages: {len(history)} | Thread ID: {thread_id[:8]}...[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error retrieving memory: {e}[/red]")


def _get_message_role(msg) -> str:
    """Extract role from message object."""
    if hasattr(msg, "type"):
        return msg.type
    if hasattr(msg, "__class__"):
        class_name = msg.__class__.__name__.lower()
        if "human" in class_name:
            return "human"
        if "ai" in class_name:
            return "ai"
        if "tool" in class_name:
            return "tool"
        if "system" in class_name:
            return "system"
    return "unknown"


def _get_message_content(msg) -> str:
    """Extract content from message object."""
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle list content (e.g., multimodal)
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(parts) if parts else "[Complex content]"
    return str(msg)


def _get_tool_info(msg) -> str:
    """Extract tool call info from message."""
    info_parts = []
    
    # Check for tool calls (AI message calling tools)
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        calls = []
        for tc in msg.tool_calls:
            name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
            calls.append(f"→ {name}")
        info_parts.append("\n".join(calls))
    
    # Check for tool call ID (tool response)
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        tool_name = getattr(msg, "name", "")
        info_parts.append(f"← {tool_name} ({msg.tool_call_id[:8]}...)")
    
    return "\n".join(info_parts) if info_parts else ""


def display_tool_call(tool_name: str, tool_args: dict):
    """Display a tool being called."""
    console.print()
    console.print(f"[yellow]🔧 Calling tool:[/yellow] [bold]{tool_name}[/bold]")
    
    if tool_args:
        # Pretty print args
        args_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        args_table.add_column("Key", style="dim")
        args_table.add_column("Value")
        
        for key, value in tool_args.items():
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            args_table.add_row(key, value_str)
        
        console.print(args_table)


def display_tool_result(tool_name: str, result: str, success: bool = True):
    """Display tool execution result."""
    status = "[green]✓[/green]" if success else "[red]✗[/red]"
    console.print(f"{status} [dim]Tool {tool_name} completed[/dim]")
    
    # Show truncated result
    if result:
        result_preview = result[:500] + "..." if len(result) > 500 else result
        console.print(Panel(
            result_preview,
            title=f"[dim]{tool_name} result[/dim]",
            border_style="dim",
            expand=False,
        ))


def display_agent_response(content: str):
    """Display the final agent response."""
    console.print()
    console.print(Panel(
        Markdown(content),
        title="[bold blue]🤖 Assistant[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))


def display_user_message(message: str):
    """Display user message."""
    console.print()
    console.print(Panel(
        message,
        title="[bold green]👤 You[/bold green]",
        border_style="green",
        padding=(0, 2),
    ))


def display_tools_list(tools: list):
    """Display available tools."""
    console.print()
    console.print(Panel("[bold]🛠️ Available Tools[/bold]", border_style="yellow"))
    
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold yellow")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description")
    
    for tool in tools:
        name = getattr(tool, "name", str(tool))
        desc = getattr(tool, "description", "No description")
        # Truncate description
        if len(desc) > 80:
            desc = desc[:80] + "..."
        table.add_row(name, desc)
    
    console.print(table)


def display_config():
    """Display current configuration."""
    from src.models.config import get_settings
    
    settings = get_settings()
    
    console.print()
    console.print(Panel("[bold]⚙️ Configuration[/bold]", border_style="magenta"))
    
    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    
    # Vision model
    table.add_row("Vision API Key", "✓ Set" if settings.is_doubao_configured else "✗ Not set")
    table.add_row("Vision Model", settings.doubao_model)
    table.add_row("Vision Base URL", settings.doubao_base_url or "default")
    
    # Agent
    table.add_row("Agent API Key", "✓ Set" if settings.is_agent_configured else "✗ Not set")
    table.add_row("Agent Model", settings.agent_model)
    table.add_row("Agent Base URL", settings.agent_base_url or "default")
    table.add_row("Agent Temperature", str(settings.agent_temperature))
    
    # Output
    table.add_row("Output Directory", str(settings.default_output_dir))
    table.add_row("Auto Save", str(settings.auto_save))
    
    console.print(table)


def display_thinking():
    """Display a thinking indicator."""
    return console.status("[bold blue]🤔 Thinking...[/bold blue]", spinner="dots")


# ============================================================================
# Streaming Chat Handler
# ============================================================================

def stream_chat(agent, message: str, thread_id: str, verbose: bool = True):
    """
    Stream agent response with tool call visualization.
    
    Uses LangGraph streaming to display:
    - Tool calls as they happen
    - Tool results
    - Final agent response
    
    Args:
        agent: The QuestionExtractionAgent instance
        message: User message to send
        thread_id: Conversation thread ID
        verbose: Whether to show detailed tool info
    """
    display_user_message(message)
    
    final_response = ""
    seen_tool_calls = set()
    seen_tool_results = set()
    
    try:
        console.print()
        console.print("[dim]Processing...[/dim]")
        
        for chunk in agent.stream(message, thread_id):
            # Process different chunk types based on LangGraph stream output
            # chunk is typically a dict with node names as keys
            
            for node_name, node_output in chunk.items():
                # Get messages from this node's output
                messages = node_output.get("messages", [])
                
                for msg in messages:
                    msg_id = id(msg)
                    role = _get_message_role(msg)
                    
                    # Handle AI messages with tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tc_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                            
                            # Only display each tool call once
                            if tc_id and tc_id not in seen_tool_calls:
                                seen_tool_calls.add(tc_id)
                                tc_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
                                tc_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                                
                                if verbose:
                                    display_tool_call(tc_name, tc_args)
                                else:
                                    console.print(f"[yellow]🔧 {tc_name}[/yellow]")
                    
                    # Handle tool result messages
                    if role == "tool" and hasattr(msg, "tool_call_id"):
                        tc_id = msg.tool_call_id
                        
                        # Only display each tool result once
                        if tc_id and tc_id not in seen_tool_results:
                            seen_tool_results.add(tc_id)
                            tool_name = getattr(msg, "name", "tool")
                            content = _get_message_content(msg)
                            
                            if verbose:
                                display_tool_result(tool_name, content)
                            else:
                                console.print(f"[green]✓[/green] [dim]{tool_name} done[/dim]")
                    
                    # Capture final AI response (no tool calls = final answer)
                    if role in ("ai", "assistant") and hasattr(msg, "content") and msg.content:
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            final_response = _get_message_content(msg)
        
        # Display final response
        if final_response:
            display_agent_response(final_response)
        
    except Exception as e:
        console.print(f"[red]Error during streaming: {e}[/red]")
        import traceback
        if os.environ.get("DEBUG"):
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        # Fall back to non-streaming
        console.print("[dim]Falling back to non-streaming mode...[/dim]")
        invoke_chat(agent, message, thread_id, verbose)


def invoke_chat(agent, message: str, thread_id: str, verbose: bool = True):
    """
    Invoke agent (non-streaming) with detailed output.
    
    Args:
        agent: The QuestionExtractionAgent instance
        message: User message to send
        thread_id: Conversation thread ID
        verbose: Whether to show detailed tool info
    """
    display_user_message(message)
    
    try:
        with display_thinking():
            result = agent.invoke(message, thread_id)
        
        # Process all messages in the result
        messages = result.get("messages", [])
        
        # Track which messages we've processed
        processed_tools = set()
        final_response = ""
        
        for msg in messages:
            role = _get_message_role(msg)
            
            # Skip the user message (we already displayed it)
            if role in ("user", "human"):
                continue
            
            # Display tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tc_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
                    tc_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
                    tc_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                    
                    if verbose:
                        display_tool_call(tc_name, tc_args)
                    else:
                        console.print(f"[yellow]🔧 {tc_name}[/yellow]")
            
            # Display tool results
            if role == "tool":
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id and tool_call_id not in processed_tools:
                    processed_tools.add(tool_call_id)
                    tool_name = getattr(msg, "name", "tool")
                    content = _get_message_content(msg)
                    
                    if verbose:
                        display_tool_result(tool_name, content)
                    else:
                        console.print(f"[green]✓[/green] [dim]{tool_name} done[/dim]")
            
            # Capture final AI response (no tool calls = final answer)
            if role in ("ai", "assistant") and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                content = _get_message_content(msg)
                if content:
                    final_response = content
        
        # Display the final response
        if final_response:
            display_agent_response(final_response)
        
    except Exception as e:
        console.print(f"[red]Error during chat: {e}[/red]")
        import traceback
        if os.environ.get("DEBUG"):
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


# ============================================================================
# CLI Commands
# ============================================================================

@click.group()
@click.version_option(version="0.1.0", prog_name="question-agent")
def main():
    """
    Question Extraction Agent - Extract questions from images using AI.
    
    This tool uses vision models to recognize questions in images and
    output them in JSON and Word document formats.
    
    Use 'question-agent interactive' for chat mode with full output.
    """
    pass


@main.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "-t", "--type",
    "question_type",
    type=click.Choice(["auto", "multiple_choice", "true_false"]),
    default="auto",
    help="Type of questions to extract"
)
@click.option(
    "-j", "--json",
    "json_output",
    type=click.Path(),
    help="Output JSON file path"
)
@click.option(
    "-w", "--word",
    "word_output",
    type=click.Path(),
    help="Output Word document path"
)
@click.option(
    "--append/--overwrite",
    default=False,
    help="Append to existing files or overwrite"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Show detailed agent output including tool calls"
)
def extract(images, question_type, json_output, word_output, append, verbose):
    """
    Extract questions from one or more images.
    
    IMAGES: One or more image file paths to process.
    
    Examples:
    
        question-agent extract image.jpg
        
        question-agent extract image1.jpg image2.jpg -j output.json
        
        question-agent extract image.jpg -t multiple_choice -j q.json -w q.docx
        
        question-agent extract image.jpg -v  # Show tool calls
    """
    from src.agent import create_question_extraction_agent
    
    console.print(Panel(
        f"[bold]Extracting questions from {len(images)} image(s)[/bold]",
        border_style="blue"
    ))
    
    # Build the request message
    image_list = ", ".join(images)
    type_hint = f"（类型：{question_type}）" if question_type != "auto" else ""
    
    request_parts = [f"请从以下图片中提取试题{type_hint}：{image_list}"]
    
    if json_output:
        mode = "追加" if append else "覆盖"
        request_parts.append(f"保存为JSON文件：{json_output}（{mode}模式）")
    
    if word_output:
        request_parts.append(f"保存为Word文档：{word_output}")
    
    if not json_output and not word_output:
        request_parts.append("请显示提取结果")
    
    request = "。".join(request_parts)
    
    # Create agent and process
    try:
        agent = create_question_extraction_agent()
        invoke_chat(agent, request, agent.thread_id, verbose)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-t", "--type",
    "question_type",
    type=click.Choice(["auto", "multiple_choice", "true_false"]),
    default="auto",
    help="Type of questions to extract"
)
@click.option(
    "-j", "--json",
    "json_output",
    type=click.Path(),
    help="Output JSON file path"
)
@click.option(
    "-w", "--word",
    "word_output",
    type=click.Path(),
    help="Output Word document path"
)
@click.option(
    "-r", "--recursive",
    is_flag=True,
    help="Process subdirectories recursively"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Show detailed agent output including tool calls"
)
def batch(directory, question_type, json_output, word_output, recursive, verbose):
    """
    Batch process all images in a directory.
    
    DIRECTORY: Directory containing image files to process.
    
    Examples:
    
        question-agent batch ./images/
        
        question-agent batch ./images/ -r -j all_questions.json
        
        question-agent batch ./images/ -v  # Show tool calls
    """
    from src.agent import create_question_extraction_agent
    
    console.print(Panel(
        f"[bold]Batch processing: {directory}[/bold]",
        border_style="blue"
    ))
    
    # Build the request message
    type_hint = f"（类型：{question_type}）" if question_type != "auto" else ""
    recursive_hint = "（包括子目录）" if recursive else ""
    
    request_parts = [f"请批量处理目录中的所有图片{type_hint}{recursive_hint}：{directory}"]
    
    if json_output:
        request_parts.append(f"将所有结果保存为JSON文件：{json_output}")
    
    if word_output:
        request_parts.append(f"将所有结果保存为Word文档：{word_output}")
    
    if not json_output and not word_output:
        request_parts.append("请显示处理统计")
    
    request = "。".join(request_parts)
    
    # Create agent and process
    try:
        agent = create_question_extraction_agent()
        invoke_chat(agent, request, agent.thread_id, verbose)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@main.command()
@click.option(
    "-l", "--language",
    type=click.Choice(["zh", "en"]),
    default="zh",
    help="Agent language (zh=Chinese, en=English)"
)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Disable streaming mode (use invoke instead)"
)
@click.option(
    "-v", "--verbose/--quiet",
    default=True,
    help="Show/hide detailed tool information"
)
def interactive(language, no_stream, verbose):
    """
    Start an interactive chat session with the agent.
    
    This mode allows conversational interaction with the question
    extraction agent for more complex workflows.
    
    Features:
    - Conversation memory (agent remembers context)
    - Tool call visualization
    - Memory inspection commands
    
    Example:
    
        question-agent interactive
        
        question-agent interactive --language en
        
        question-agent interactive --quiet  # Hide tool details
    """
    from src.agent import create_question_extraction_agent
    
    console.clear()
    display_welcome()
    
    try:
        agent = create_question_extraction_agent(language=language)
        thread_id = agent.thread_id
        
        console.print(f"\n[dim]Session started. Thread ID: {thread_id[:8]}...[/dim]")
        console.print(f"[dim]Language: {'Chinese' if language == 'zh' else 'English'} | Verbose: {verbose}[/dim]\n")
        
    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {e}[/red]")
        console.print("[yellow]Please check your API keys in .env file.[/yellow]")
        return
    
    # Main chat loop
    while True:
        try:
            # Get user input
            console.print()
            user_input = console.input("[bold green]You>[/bold green] ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.startswith("/"):
                cmd_parts = user_input.lower().split()
                cmd = cmd_parts[0]
                
                if cmd in ("/exit", "/quit", "/q"):
                    console.print("\n[blue]Goodbye! 👋[/blue]")
                    break
                
                elif cmd == "/help":
                    display_welcome()
                    continue
                
                elif cmd == "/memory":
                    display_memory(agent, thread_id)
                    continue
                
                elif cmd == "/clear":
                    thread_id = agent.new_conversation()
                    console.print(f"[green]✓ Started new conversation. Thread ID: {thread_id[:8]}...[/green]")
                    continue
                
                elif cmd == "/config":
                    display_config()
                    continue
                
                elif cmd == "/tools":
                    display_tools_list(agent.tools)
                    continue
                
                elif cmd == "/verbose":
                    verbose = not verbose
                    console.print(f"[green]✓ Verbose mode: {'ON' if verbose else 'OFF'}[/green]")
                    continue
                
                elif cmd == "/debug":
                    # Toggle debug mode
                    if os.environ.get("DEBUG"):
                        del os.environ["DEBUG"]
                        console.print("[green]✓ Debug mode: OFF[/green]")
                    else:
                        os.environ["DEBUG"] = "1"
                        console.print("[green]✓ Debug mode: ON[/green]")
                    continue
                
                else:
                    console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]")
                    continue
            
            # Regular chat message
            if no_stream:
                invoke_chat(agent, user_input, thread_id, verbose)
            else:
                # Try streaming first, fall back to invoke
                try:
                    stream_chat(agent, user_input, thread_id, verbose)
                except NotImplementedError:
                    console.print(f"[dim]Streaming not available, using invoke mode...[/dim]")
                    invoke_chat(agent, user_input, thread_id, verbose)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit[/yellow]")
            continue
        
        except EOFError:
            console.print("\n[blue]Goodbye! 👋[/blue]")
            break


@main.command()
def config():
    """
    Display current configuration settings.
    
    Shows the current configuration loaded from environment variables
    and .env file.
    """
    display_config()


@main.command()
def tools():
    """
    List all available agent tools.
    
    Shows the tools that the agent can use to process images
    and generate output files.
    """
    from src.tools import get_all_tools
    
    display_tools_list(get_all_tools())


if __name__ == "__main__":
    main()
