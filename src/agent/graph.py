"""
LangGraph entry point for LangSmith Studio.

This file exposes the compiled graph for LangSmith Studio to visualize and test.
"""

from src.agent.agent import QuestionExtractionAgent

# Create the agent instance
# We use default settings which will load from .env
# We disable the default checkpointer because LangGraph Studio handles persistence automatically
agent_instance = QuestionExtractionAgent(checkpointer=None)

# Expose the compiled graph
# The agent attribute in QuestionExtractionAgent is the compiled graph
graph = agent_instance.agent
