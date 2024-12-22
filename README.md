# Ollama Model Benchmarker

A comprehensive toolkit for evaluating and comparing Ollama models in real-world scenarios.

## Purpose

This tool helps you:
- Compare different Ollama models' performance
- Measure response times and token generation speeds
- Test models under various loads (sequential vs parallel)
- Monitor system resource usage during inference
- Evaluate model responses across different types of prompts

## Features

### Performance Testing
- Response time measurements
- Token generation speed
- Streaming vs non-streaming comparison
- Parallel processing capabilities
- Connection and resource monitoring

### Model Comparison
- Side-by-side model outputs
- Response quality metrics
- Memory and CPU usage
- Chunk size analysis
- Token efficiency

### Testing Scenarios
- Basic prompts
- Complex reasoning
- Code generation
- Creative writing
- Mathematical problems

## Installation

```bash
poetry install
```

## Usage

### Basic Model Testing
```python
from ollama_model_benchmarker import *

# Single model test
bot = OllamaBot()
response = bot.generate_response(
    prompt="Explain quantum computing",
    stream=True  # Enable streaming for token-by-token analysis
)
```

### Comparative Testing
```python
# Sequential comparison
run_sequential_queries([
    "llama3.2",
    "codellama",
    "nemotron-mini"
], prompt="Write a Python function that...")

# Parallel comparison
await run_parallel_queries([
    "llama3.2",
    "codellama",
    "nemotron-mini"
], prompt="Solve this math problem...")
```

### Performance Metrics
```python
# Get detailed metrics
metrics = bot.get_performance_metrics()
print(f"Response Time: {metrics['response_time']}s")
print(f"Tokens/second: {metrics['tokens_per_second']}")
print(f"Memory Usage: {metrics['memory_usage']}MB")
```

## Development

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest test_program.py -k "test_streaming"
poetry run pytest test_program.py -k "test_parallel"
```

## Contributing

Contributions welcome! Areas of interest:
- Additional model support
- New testing scenarios
- Performance optimizations
- Metric collection enhancements

## License

MIT
