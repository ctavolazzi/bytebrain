# NovaSystem Benchmarking

A modular benchmarking system for LLM models using Ollama.

## Features

- Benchmark multiple Ollama models in parallel or sequential mode
- Web interface for running and viewing benchmarks
- Detailed performance metrics including:
  - Time to first token
  - Total processing time
  - Tokens per second
  - System resource usage
- CLI tool for easy server management
- Async client for Ollama API

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NovaSystem_Benchmarking.git
cd NovaSystem_Benchmarking

# Install with Poetry
poetry install
```

## Usage

### Starting the Benchmark Server

```bash
# Start the server with auto-reload
poetry run novasystem-benchmark serve --reload

# Start on a specific port
poetry run novasystem-benchmark serve --port 8080

# Start with custom host
poetry run novasystem-benchmark serve --host 127.0.0.1
```

### Using the Web Interface

1. Start the server
2. Visit http://localhost:8000/api/benchmarks/
3. Select models to benchmark
4. Enter a prompt
5. Click "Run Benchmark"

### Using the Python API

```python
from novasystem_benchmarking import OllamaClient
import asyncio

async def main():
    # Initialize the client
    client = OllamaClient()

    # List available models
    models = await client.list_models()
    print("Available models:", models)

    # Run a chat completion
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    async for response in client.chat("llama2", messages):
        if "response" in response:
            print(response["response"], end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

## Development

### Setup

```bash
# Install development dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black novasystem_benchmarking
poetry run isort novasystem_benchmarking

# Type checking
poetry run mypy novasystem_benchmarking
```

### Project Structure

```
novasystem_benchmarking/
├── data/           # Benchmark results storage
├── models/         # Pydantic models
├── routers/        # FastAPI routes
├── services/       # Core services
├── static/         # Static files
├── templates/      # HTML templates
├── __init__.py    # Package initialization
└── cli.py         # Command-line interface
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details
