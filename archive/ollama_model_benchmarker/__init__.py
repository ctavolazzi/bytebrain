"""
Ollama Model Benchmarker - A simple interface to Ollama models
"""

from bots.ollama_bot import OllamaBot
from bots.model_comparison import compare_responses, load_config

__version__ = "0.1.0"
__author__ = "Christopher Tavolazzi"

__all__ = ['OllamaBot', 'compare_responses', 'load_config']