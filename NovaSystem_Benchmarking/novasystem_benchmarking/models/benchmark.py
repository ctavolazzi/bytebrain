from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class BenchmarkRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to test with the models")
    models: List[str] = Field(default=["wizardlm2", "nemotron-mini", "llama3.2"],
                            description="List of Ollama models to benchmark")
    parameters: Optional[Dict] = Field(default=None,
                                     description="Optional parameters for model generation")
    parallel_processing: bool = True  # Default to parallel processing

class SystemInfo(BaseModel):
    platform: str = Field(default="Unknown")
    processor: str = Field(default="Unknown")
    python_version: str = Field(default="Unknown")
    cpu: Dict = Field(default_factory=lambda: {"physical_cores": 0, "total_cores": 0})
    memory: Dict = Field(default_factory=lambda: {"total": 0, "available": 0, "used": 0, "percent_used": 0})
    gpu: Optional[List[Dict]] = None

class BenchmarkResult(BaseModel):
    model: str = Field(default="Unknown Model")
    timing: Dict = Field(default_factory=dict)
    throughput: Dict = Field(default_factory=dict)
    system_impact: Dict = Field(default_factory=dict)
    success: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response: Optional[str] = None

class BenchmarkResponse(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    system_info: SystemInfo = Field(default_factory=SystemInfo)
    prompt: str = Field(default="Unknown prompt", description="The prompt used for all models in this benchmark")
    processing_mode: str = Field(default="parallel", description="Whether the benchmark was run in 'parallel' or 'sequential' mode")
    results: List[BenchmarkResult] = Field(default_factory=list)