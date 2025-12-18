"""
Configuration management using Pydantic Settings.

This module provides centralized configuration for the question extraction agent,
loading values from environment variables and .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden by environment variables.
    See .env.example for documentation of each setting.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # -------------------------------------------------------------------------
    # Vision Model Configuration (Doubao)
    # -------------------------------------------------------------------------
    doubao_api_key: str = Field(
        default="",
        description="API key for Doubao vision model"
    )
    doubao_base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="Doubao API base URL"
    )
    doubao_model: str = Field(
        default="doubao-seed-1-6-251015",
        description="Doubao model endpoint ID"
    )
    doubao_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for vision model response"
    )
    doubao_temperature: float = Field(
        default=0.1,
        description="Temperature for vision model"
    )
    
    # -------------------------------------------------------------------------
    # Agent LLM Configuration
    # -------------------------------------------------------------------------
    agent_api_key: Optional[str] = Field(
        default=None,
        description="API key for agent LLM (defaults to DOUBAO_API_KEY if not set)"
    )
    agent_model: str = Field(
        default="doubao-seed-1-6-lite-251015",
        description="Model for agent reasoning"
    )
    agent_base_url: Optional[str] = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="Custom base URL for OpenAI-compatible APIs"
    )
    agent_max_iterations: int = Field(
        default=10,
        description="Maximum iterations for agent execution"
    )
    agent_verbose: bool = Field(
        default=True,
        description="Enable verbose logging for agent"
    )
    agent_enable_memory: bool = Field(
        default=True,
        description="Enable conversation memory"
    )
    agent_temperature: float = Field(
        default=0.0,
        description="Temperature for agent LLM"
    )
    
    agent_provider: str = Field(
        default="openai",
        description="Provider for agent LLM (openai, anthropic)"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="API key for Anthropic"
    )

    # -------------------------------------------------------------------------
    # Output Configuration
    # -------------------------------------------------------------------------
    default_output_dir: Path = Field(
        default=Path("./output"),
        description="Default directory for output files"
    )
    default_json_filename: str = Field(
        default="questions.json",
        description="Default filename for JSON output"
    )
    default_word_filename: str = Field(
        default="questions.docx",
        description="Default filename for Word output"
    )
    auto_save: bool = Field(
        default=True,
        description="Automatically save extracted questions"
    )
    append_by_default: bool = Field(
        default=False,
        description="Append to existing files by default"
    )
    json_indent: int = Field(
        default=2,
        description="JSON indentation spaces"
    )
    
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Log level"
    )
    
    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @property
    def is_doubao_configured(self) -> bool:
        """Check if Doubao API is properly configured."""
        return bool(self.doubao_api_key)
    
    @property
    def is_agent_configured(self) -> bool:
        """Check if agent LLM is properly configured."""
        return bool(self.agent_api_key or self.doubao_api_key)
    
    @property
    def effective_agent_api_key(self) -> str:
        """Get the effective API key for the agent (falls back to Doubao key)."""
        return self.agent_api_key or self.doubao_api_key or ""
    
    def get_output_path(self, filename: Optional[str] = None, file_type: str = "json") -> Path:
        """
        Get the full output path for a file.
        
        Args:
            filename: Optional custom filename
            file_type: Type of file ('json' or 'word')
            
        Returns:
            Full path to the output file
        """
        if filename:
            return self.default_output_dir / filename
        
        if file_type == "json":
            return self.default_output_dir / self.default_json_filename
        elif file_type == "word":
            return self.default_output_dir / self.default_word_filename
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    def ensure_output_dir(self) -> Path:
        """Ensure the output directory exists and return its path."""
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        return self.default_output_dir


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings (cached singleton).
    
    Returns:
        Settings instance loaded from environment
    """
    return Settings()
