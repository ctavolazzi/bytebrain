"""Pydantic models for Autogen benchmark requests and responses."""
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Configuration for an LLM backend."""
    provider: Literal["openai", "anthropic", "local", "azure"] = "openai"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    additional_config: Optional[Dict[str, Any]] = None

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

class BenchmarkRequest(BaseModel):
    """Request model for running benchmarks."""
    prompt: str
    configs: List[ConversationConfig]
    parallel_processing: bool = False
    parameters: Optional[Dict[str, Any]] = None

class SystemMetrics(BaseModel):
    """System metrics during benchmark."""
    cpu_delta: List[float]
    memory_delta: float

class TimingMetrics(BaseModel):
    """Timing metrics for benchmark results."""
    total_time: float
    error: Optional[str] = None

class ConversationMetrics(BaseModel):
    """Metrics specific to conversation performance."""
    messages_sent: int
    total_tokens: int
    average_response_time: float
    messages: List[Dict[str, Any]]

class BenchmarkResult(BaseModel):
    """Result model for a single conversation benchmark."""
    config_name: str
    success: bool
    timing: TimingMetrics
    metrics: Optional[ConversationMetrics] = None
    system_impact: Optional[SystemMetrics] = None
    timestamp: datetime
    error: Optional[str] = None

class SystemInfo(BaseModel):
    """System information during benchmark."""
    platform: str
    python_version: str
    cpu: Dict[str, Any]
    memory: Dict[str, float]
    timestamp: datetime

class BenchmarkResponse(BaseModel):
    """Response model for benchmark results."""
    prompt: str
    processing_mode: str
    parameters: Optional[Dict[str, Any]] = None
    system_info: SystemInfo
    results: List[BenchmarkResult]
    timestamp: datetime