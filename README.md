# Ollama Model Benchmarker

A FastAPI-based application for benchmarking local Large Language Models (LLMs) via Ollama. This tool helps you compare different models' performance, resource usage, and response quality.

## Features

- Benchmark multiple Ollama models in parallel
- Collect detailed system metrics (CPU, memory, GPU usage)
- Track timing metrics (time to first token, total time)
- Store results in MongoDB (with JSON file fallback)
- Simple web interface for running benchmarks and viewing results

## Prerequisites

- Python 3.10 or higher
- Ollama installed and running locally
- MongoDB (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ollama-benchmarker
```

2. Install dependencies using Poetry:
```bash
poetry install
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Configuration

The application uses environment variables for configuration:

- `MONGODB_URL`: MongoDB connection URL (default: "mongodb://localhost:27017")
- `PORT`: Port to run the FastAPI server (default: 8000)

Create a `.env` file in the project root:
```
MONGODB_URL=mongodb://localhost:27017
PORT=8000
```

## Usage

1. Start the FastAPI server:
```bash
poetry run uvicorn app.main:app --reload
```

2. Open your browser and navigate to:
```
http://localhost:8000/api/benchmarks/
```

3. Enter a prompt and select the models you want to benchmark.

4. View the results in real-time, including:
   - System information
   - Response timing
   - Resource usage
   - Model outputs

## API Endpoints

- `POST /api/benchmarks/run`: Run benchmarks
  ```json
  {
    "prompt": "Your prompt here",
    "models": ["wizardlm2", "nemotron-mini"],
    "parameters": {}
  }
  ```

- `GET /api/benchmarks/history`: Get benchmark history
- `GET /api/benchmarks/history/{id}`: Get specific benchmark result
- `GET /api/benchmarks/`: Web UI

## Development

1. Install development dependencies:
```bash
poetry install --with dev
```

2. Run tests:
```bash
poetry run pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
