import pytest
import subprocess
import json
from app.services.benchmark import BenchmarkService

def ensure_model_exists(model_name: str):
    """Ensure a model is pulled before testing."""
    try:
        # Check if model exists
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True
        )
        models = result.stdout.lower()

        if model_name not in models:
            print(f"Pulling model {model_name}...")
            subprocess.run(
                ['ollama', 'pull', model_name],
                check=True
            )
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to pull model {model_name}: {e}")
    except FileNotFoundError:
        pytest.skip("Ollama CLI not found. Please install Ollama first.")

@pytest.mark.asyncio
async def test_benchmark_service_with_real_ollama():
    """Test the benchmark service with a real Ollama model."""
    model_name = "llama3.2"  # Using llama3.2 as we have it available
    ensure_model_exists(model_name)

    service = BenchmarkService()

    # Test with a simple prompt and one model
    result = await service.run_benchmark(
        prompt="Respond in exactly one sentence: How are you?",
        models=[model_name]
    )

    # Verify system info
    assert result.system_info is not None
    assert result.system_info.platform is not None
    assert result.system_info.cpu is not None
    assert result.system_info.memory is not None

    # Verify benchmark results
    assert len(result.results) == 1
    model_result = result.results[0]

    # Basic result structure
    assert model_result.model == model_name
    assert model_result.prompt == "Respond in exactly one sentence: How are you?"
    assert model_result.success is True

    # Timing checks
    assert model_result.timing["time_to_first_token"] > 0
    assert model_result.timing["total_time"] > 0

    # Throughput checks
    assert model_result.throughput["total_chunks"] > 0
    assert model_result.throughput["total_bytes"] > 0
    assert model_result.throughput["average_chunk_size"] > 0
    assert model_result.throughput["bytes_per_second"] > 0

    # System impact checks
    assert isinstance(model_result.system_impact["cpu_delta"], list)
    assert isinstance(model_result.system_impact["memory_delta"], float)

    # Response checks
    assert model_result.response is not None
    assert len(model_result.response) > 0
    assert model_result.response.count('.') == 1  # Verify it's one sentence

@pytest.mark.asyncio
async def test_parallel_benchmark_execution():
    """Test running multiple models in parallel."""
    models = ["llama3.2", "nemotron-mini"]  # Models we have available
    for model in models:
        ensure_model_exists(model)

    service = BenchmarkService()

    # Test with multiple models
    result = await service.run_benchmark(
        prompt="Respond in exactly one sentence: What is 2+2?",
        models=models
    )

    # Verify we got results for both models
    assert len(result.results) == 2

    # Check each model's results
    for model_result in result.results:
        assert model_result.success is True
        assert model_result.timing["total_time"] > 0
        assert model_result.response is not None
        assert model_result.response.count('.') == 1  # Verify it's one sentence

@pytest.mark.asyncio
async def test_benchmark_error_handling():
    """Test error handling with an invalid model."""
    service = BenchmarkService()

    # Test with a non-existent model
    result = await service.run_benchmark(
        prompt="Respond in exactly one sentence: Hi.",
        models=["non_existent_model_123"]
    )

    assert len(result.results) == 1
    model_result = result.results[0]

    assert model_result.success is False
    assert "error" in model_result.timing
    assert not model_result.response