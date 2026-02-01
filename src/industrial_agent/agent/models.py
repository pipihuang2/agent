"""Model configuration for DeepSeek and Ollama backends."""

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel

from industrial_agent.config import Settings, get_settings
from industrial_agent.config.settings import ModelProvider


def get_model(settings: Settings | None = None) -> Model:
    """
    Get the appropriate model instance based on configuration.

    Args:
        settings: Optional settings instance. If not provided, uses default settings.

    Returns:
        A pydantic-ai Model instance configured for the selected provider.

    Raises:
        ValueError: If the model provider is not supported or configuration is invalid.
    """
    if settings is None:
        settings = get_settings()

    settings.validate_config()

    if settings.model_provider == ModelProvider.DEEPSEEK:
        return _create_deepseek_model(settings)
    elif settings.model_provider == ModelProvider.OLLAMA:
        return _create_ollama_model(settings)
    else:
        raise ValueError(f"Unsupported model provider: {settings.model_provider}")


def _create_deepseek_model(settings: Settings) -> Model:
    """Create a DeepSeek model instance."""
    # DeepSeek uses OpenAI-compatible API
    return OpenAIModel(
        model_name=settings.model_name,
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
    )


def _create_ollama_model(settings: Settings) -> Model:
    """Create an Ollama model instance."""
    # Ollama also provides OpenAI-compatible API
    return OpenAIModel(
        model_name=settings.model_name,
        base_url=f"{settings.ollama_base_url}/v1",
        api_key="ollama",  # Ollama doesn't require a real API key
    )
