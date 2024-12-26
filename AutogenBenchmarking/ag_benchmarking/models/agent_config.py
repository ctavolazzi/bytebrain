"""Shared models for agent configuration."""
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

class OllamaConfig(BaseModel):
    """Ollama-specific configuration."""
    model: str = "wizardlm2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    context_window: Optional[int] = None
    timeout: Optional[float] = None

class LLMConfig(BaseModel):
    """Configuration for an LLM backend."""
    provider: Literal["openai", "anthropic", "local", "azure", "ollama"] = "ollama"
    config_list: List[Dict[str, Any]] = Field(default_factory=list)
    temperature: float = 0.7
    timeout: Optional[float] = None
    cache_seed: Optional[int] = None

    @classmethod
    def create_ollama_config(cls, ollama_config: OllamaConfig) -> "LLMConfig":
        """Create an LLM config for Ollama."""
        return cls(
            provider="ollama",
            config_list=[{
                "model": ollama_config.model,
                "base_url": ollama_config.base_url,
                "api_type": "ollama",
                "temperature": ollama_config.temperature,
                "context_window": ollama_config.context_window,
                "timeout": ollama_config.timeout
            }]
        )

class AgentConfig(BaseModel):
    """Configuration for an Autogen agent."""
    name: str
    type: str  # assistant, user_proxy, etc.
    llm_config: Optional[LLMConfig] = None
    system_message: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)

class ConversationConfig(BaseModel):
    """Configuration for an Autogen conversation."""
    name: str
    agents: List[AgentConfig]
    initiator: str  # Name of the agent that starts the conversation
    max_rounds: int = Field(default=10, ge=1)
    description: Optional[str] = None