"""
NovaSystem - A sophisticated local LLM bot system using Autogen and Ollama.

This package provides a complete system for building and managing conversations
with local LLM models through Ollama, featuring:

- Detailed conversation logging and metadata tracking
- Multi-agent orchestration (Planner, Executor, Memory)
- Streaming and non-streaming response support
- System metrics collection
- Session management
"""

from .core import NovaSystemCore
from .agents import (
    AgentOrchestrator,
    PlannerAgent,
    ExecutorAgent,
    MemoryAgent
)
from .interface import NovaSystem

__version__ = "0.1.0"
__author__ = "ByteBrain Team"

__all__ = [
    "NovaSystem",
    "NovaSystemCore",
    "AgentOrchestrator",
    "PlannerAgent",
    "ExecutorAgent",
    "MemoryAgent"
]