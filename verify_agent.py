"""
Agent Verification Script

This script verifies that the Question Extraction Agent is working correctly
by testing it with sample images (2.jpg and 7.jpg).

Usage:
    python verify_agent.py
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def print_separator(title: str = "", char: str = "=", length: int = 70):
    """Print a separator line with optional title."""
    if title:
        padding = (length - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * length)


def print_step(step_num: int, description: str):
    """Print a step header."""
    print(f"\n{'─' * 70}")
    print(f"📌 Step {step_num}: {description}")
    print(f"{'─' * 70}")


def verify_environment():
    """Verify the environment is correctly set up."""
    print_separator("Environment Verification", "=")
    
    logger.info("Checking Python version...")
    logger.info(f"  Python: {sys.version}")
    
    # Check required packages
    logger.info("Checking required packages...")
    required_packages = [
        ("langchain", "langchain"),
        ("langchain_openai", "langchain-openai"),
        ("langgraph", "langgraph"),
        ("openai", "openai"),
        ("pydantic", "pydantic"),
        ("docx", "python-docx"),
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            logger.info(f"  ✅ {package_name}")
        except ImportError:
            logger.error(f"  ❌ {package_name} - NOT INSTALLED")
            missing.append(package_name)
    
    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    # Check test images
    logger.info("Checking test images...")
    test_images = ["2.jpg", "7.jpg"]
    for img in test_images:
        img_path = Path(img)
        if img_path.exists():
            size_kb = img_path.stat().st_size / 1024
            logger.info(f"  ✅ {img} ({size_kb:.1f} KB)")
        else:
            logger.error(f"  ❌ {img} - NOT FOUND")
            return False
    
    # Check .env file
    logger.info("Checking configuration...")
    env_path = Path(".env")
    if env_path.exists():
        logger.info("  ✅ .env file found")
    else:
        logger.warning("  ⚠️ .env file not found - will use environment variables")
    
    return True


def verify_settings():
    """Verify settings are loaded correctly."""
    print_separator("Settings Verification", "=")
    
    logger.info("Loading settings from configuration...")
    try:
        from src.models.config import get_settings
        settings = get_settings()
        
        logger.info(f"  Doubao Model: {settings.doubao_model}")
        logger.info(f"  Doubao Base URL: {settings.doubao_base_url or 'default'}")
        logger.info(f"  Agent Model: {settings.agent_model}")
        logger.info(f"  Agent Base URL: {settings.agent_base_url or 'default'}")
        logger.info(f"  Agent Temperature: {settings.agent_temperature}")
        
        # Check API key (masked)
        if settings.doubao_api_key:
            key = settings.doubao_api_key
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            logger.info(f"  Doubao API Key: {masked}")
        else:
            logger.error("  ❌ Doubao API Key not configured!")
            return False
            
        if settings.effective_agent_api_key:
            key = settings.effective_agent_api_key
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            logger.info(f"  Agent API Key: {masked}")
        else:
            logger.error("  ❌ Agent API Key not configured!")
            return False
        
        logger.info("  ✅ Settings loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Failed to load settings: {e}")
        return False


def verify_tools():
    """Verify all tools are available."""
    print_separator("Tools Verification", "=")
    
    logger.info("Loading agent tools...")
    try:
        from src.tools import get_all_tools
        tools = get_all_tools()
        
        logger.info(f"  Found {len(tools)} tools:")
        for tool in tools:
            tool_name = getattr(tool, 'name', str(tool))
            logger.info(f"    • {tool_name}")
        
        logger.info("  ✅ All tools loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Failed to load tools: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_agent_creation():
    """Verify agent can be created."""
    print_separator("Agent Creation Verification", "=")
    
    logger.info("Creating Question Extraction Agent...")
    try:
        from src.agent import QuestionExtractionAgent
        
        start_time = time.time()
        agent = QuestionExtractionAgent()
        elapsed = time.time() - start_time
        
        logger.info(f"  ✅ Agent created successfully in {elapsed:.2f}s")
        logger.info(f"  Thread ID: {agent.thread_id}")
        logger.info(f"  Tools count: {len(agent.tools)}")
        
        return agent
        
    except Exception as e:
        logger.error(f"  ❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_image_analysis(agent, image_path: str):
    """Test image analysis on a single image."""
    print_step(1, f"Testing Image Analysis - {image_path}")
    
    logger.info(f"Sending request to analyze '{image_path}'...")
    logger.info("Request: 请分析这张图片中的题目: " + image_path)
    
    try:
        start_time = time.time()
        
        # Use streaming to show progress
        logger.info("Waiting for agent response (this may take a while)...")
        
        response = agent.chat(f"请分析图片 {image_path} 中的题目，提取所有题目内容")
        
        elapsed = time.time() - start_time
        logger.info(f"Response received in {elapsed:.2f}s")
        
        print("\n📝 Agent Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        return True, response
        
    except Exception as e:
        logger.error(f"❌ Image analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_save_to_json(agent, output_path: str = "verify_output.json"):
    """Test saving questions to JSON."""
    print_step(2, "Testing Save to JSON")
    
    logger.info(f"Requesting to save questions to '{output_path}'...")
    
    try:
        start_time = time.time()
        response = agent.chat(f"请将刚才提取的题目保存到JSON文件 {output_path}")
        elapsed = time.time() - start_time
        
        logger.info(f"Response received in {elapsed:.2f}s")
        
        print("\n📝 Agent Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Check if file was created
        if Path(output_path).exists():
            logger.info(f"✅ JSON file created: {output_path}")
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"   File size: {len(content)} bytes")
        else:
            logger.warning(f"⚠️ JSON file not found: {output_path}")
        
        return True, response
        
    except Exception as e:
        logger.error(f"❌ Save to JSON failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_save_to_word(agent, output_path: str = "verify_output.docx"):
    """Test saving questions to Word document."""
    print_step(3, "Testing Save to Word Document")
    
    logger.info(f"Requesting to save questions to '{output_path}'...")
    
    try:
        start_time = time.time()
        response = agent.chat(f"请将题目保存到Word文档 {output_path}")
        elapsed = time.time() - start_time
        
        logger.info(f"Response received in {elapsed:.2f}s")
        
        print("\n📝 Agent Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Check if file was created
        if Path(output_path).exists():
            logger.info(f"✅ Word file created: {output_path}")
            size_kb = Path(output_path).stat().st_size / 1024
            logger.info(f"   File size: {size_kb:.1f} KB")
        else:
            logger.warning(f"⚠️ Word file not found: {output_path}")
        
        return True, response
        
    except Exception as e:
        logger.error(f"❌ Save to Word failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_batch_processing(agent, images: list):
    """Test batch processing of multiple images."""
    print_step(4, "Testing Batch Processing")
    
    images_str = ", ".join(images)
    logger.info(f"Requesting to process multiple images: {images_str}")
    
    try:
        start_time = time.time()
        
        # Start a new conversation for batch processing
        agent.new_conversation()
        logger.info(f"Started new conversation: {agent.thread_id}")
        
        response = agent.chat(
            f"请批量处理以下图片中的题目: {images_str}，提取所有题目并保存到 batch_output.json"
        )
        elapsed = time.time() - start_time
        
        logger.info(f"Response received in {elapsed:.2f}s")
        
        print("\n📝 Agent Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        return True, response
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_conversation_memory(agent):
    """Test that the agent remembers previous conversation."""
    print_step(5, "Testing Conversation Memory")
    
    logger.info("Testing if agent remembers previous context...")
    
    try:
        # Ask about the previous extraction without specifying the image
        response = agent.chat("刚才我们处理了哪些图片？提取了多少道题目？")
        
        print("\n📝 Agent Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Get conversation history
        history = agent.get_conversation_history()
        logger.info(f"Conversation history contains {len(history)} messages")
        
        return True, response
        
    except Exception as e:
        logger.error(f"❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def run_verification():
    """Run the full verification suite."""
    print_separator("Question Extraction Agent Verification", "═")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {Path.cwd()}")
    
    results = {
        "environment": False,
        "settings": False,
        "tools": False,
        "agent_creation": False,
        "image_analysis": False,
        "save_json": False,
        "save_word": False,
        "batch_processing": False,
        "memory": False,
    }
    
    # Step 1: Verify environment
    results["environment"] = verify_environment()
    if not results["environment"]:
        logger.error("Environment verification failed. Please fix the issues above.")
        return results
    
    # Step 2: Verify settings
    results["settings"] = verify_settings()
    if not results["settings"]:
        logger.error("Settings verification failed. Please check your .env file.")
        return results
    
    # Step 3: Verify tools
    results["tools"] = verify_tools()
    if not results["tools"]:
        logger.error("Tools verification failed.")
        return results
    
    # Step 4: Create agent
    agent = verify_agent_creation()
    results["agent_creation"] = agent is not None
    if not results["agent_creation"]:
        logger.error("Agent creation failed.")
        return results
    
    # Step 5: Test image analysis with first image
    success, _ = test_image_analysis(agent, "2.jpg")
    results["image_analysis"] = success
    
    if success:
        # Step 6: Test save to JSON
        success, _ = test_save_to_json(agent)
        results["save_json"] = success
        
        # Step 7: Test save to Word
        success, _ = test_save_to_word(agent)
        results["save_word"] = success
    
    # Step 8: Test batch processing with both images
    success, _ = test_batch_processing(agent, ["2.jpg", "7.jpg"])
    results["batch_processing"] = success
    
    # Step 9: Test conversation memory
    success, _ = test_conversation_memory(agent)
    results["memory"] = success
    
    # Print summary
    print_separator("Verification Summary", "═")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All verifications passed! The agent is working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the logs above.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


def main():
    """Main entry point."""
    try:
        results = run_verification()
        
        # Exit with appropriate code
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Verification interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
