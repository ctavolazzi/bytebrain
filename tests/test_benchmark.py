import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.benchmark import BenchmarkRequest

client = TestClient(app)

def test_benchmark_endpoint():
    """Test the benchmark endpoint with a simple request."""
    request_data = {
        "prompt": "Test prompt",
        "models": ["wizardlm2"],
        "parameters": None
    }

    response = client.post("/api/benchmarks/run", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "system_info" in data
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["model"] == "wizardlm2"
    assert data["results"][0]["success"] is True

def test_invalid_request():
    """Test the benchmark endpoint with invalid data."""
    # Missing prompt
    request_data = {
        "models": ["wizardlm2"]
    }

    response = client.post("/api/benchmarks/run", json=request_data)
    assert response.status_code == 422  # Validation error

def test_benchmark_ui():
    """Test that the UI endpoint returns HTML."""
    response = client.get("/api/benchmarks/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

@pytest.mark.asyncio
async def test_benchmark_history():
    """Test the benchmark history endpoint."""
    response = client.get("/api/benchmarks/history")
    assert response.status_code == 200

    # For now, it should return the placeholder message
    data = response.json()
    assert "message" in data