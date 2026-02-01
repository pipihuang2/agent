"""Configuration management using pydantic-settings."""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(str, Enum):
    """Supported model providers."""

    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model Provider Configuration
    model_provider: ModelProvider = Field(
        default=ModelProvider.DEEPSEEK,
        description="Model provider to use (deepseek or ollama)",
    )
    model_name: str = Field(
        default="deepseek-chat",
        description="Name of the model to use",
    )

    # DeepSeek Configuration
    deepseek_api_key: Optional[str] = Field(
        default=None,
        description="DeepSeek API key",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API base URL",
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )

    # MCP Server Configuration
    mcp_server_host: str = Field(
        default="localhost",
        description="MCP server host",
    )
    mcp_server_port: int = Field(
        default=8000,
        description="MCP server port",
    )

    # Chart Output Configuration
    chart_output_dir: Path = Field(
        default=Path("./output/charts"),
        description="Directory for chart output",
    )

    def validate_config(self) -> None:
        """Validate configuration based on selected provider."""
        if self.model_provider == ModelProvider.DEEPSEEK:
            if not self.deepseek_api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY is required when using DeepSeek provider"
                )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
