# AutoGen Benchmarking System

A comprehensive benchmarking system for AutoGen agents that measures performance, resource usage, and conversation metrics.

## Features

- Benchmark multiple agent configurations
- Measure system resource usage (CPU, memory, disk, network)
- Track conversation metrics (messages, tokens, response times)
- Support for parallel and sequential processing
- REST API with real-time WebSocket updates
- Command-line interface
- Configurable via YAML or JSON

## Installation

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

## Usage

### Command Line Interface

1. Create a sample configuration:
```bash
poetry run python -m ag_benchmarking.cli create-config
```

2. Run a benchmark:
```bash
poetry run python -m ag_benchmarking.cli run benchmark_config.yaml "What is the capital of France?"
```

3. View benchmark history:
```bash
poetry run python -m ag_benchmarking.cli history ./data/benchmarks
```

### REST API

1. Start the API server:
```bash
poetry run uvicorn ag_benchmarking.api:app --reload
```

2. Run a benchmark:
```bash
curl -X POST http://localhost:8000/api/benchmarks/run \
  -H "Content-Type: application/json" \
  -d @benchmark_request.json
```

3. View benchmark history:
```bash
curl http://localhost:8000/api/benchmarks/history
```

## Configuration

Create a YAML file with your benchmark configuration:

```yaml
configurations:
  - name: basic_conversation
    description: Basic conversation between assistant and user
    agents:
      - name: assistant
        type: assistant
        llm_config:
          provider: openai
          model: gpt-3.5-turbo
          api_key: ${OPENAI_API_KEY}
        system_message: You are a helpful AI assistant.
      - name: user
        type: user_proxy
        system_message: You are a user seeking assistance.
    initiator: user
    max_rounds: 5
```

## Metrics

The benchmarking system collects the following metrics:

### System Metrics
- CPU usage per core
- Memory usage
- Disk usage
- Network I/O

### Conversation Metrics
- Number of messages sent
- Total tokens used
- Average response time
- Full conversation history

### Timing Metrics
- Total execution time
- Time to first response
- Average time between messages

## Development

1. Install development dependencies:
```bash
poetry install --with dev
```

2. Run tests:
```bash
poetry run pytest tests/
```

3. Run linter:
```bash
poetry run flake8 ag_benchmarking/
```

4. Format code:
```bash
poetry run black ag_benchmarking/
poetry run isort ag_benchmarking/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License