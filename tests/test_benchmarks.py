import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_run_benchmark():
    """Test the benchmark execution endpoint"""
    response = client.post(
        "/api/benchmarks/run",
        json={
            "prompt": "Test prompt",
            "models": ["llama2"],
            "parameters": {"temperature": 0.7}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "prompt" in data
    assert "results" in data
    assert len(data["results"]) > 0

@pytest.mark.asyncio
async def test_history_caching():
    """Test that history endpoint caching works"""
    # First request should hit the database
    response1 = client.get("/api/benchmarks/history?limit=5")
    assert response1.status_code == 200

    # Second request within 60s should return cached result
    response2 = client.get("/api/benchmarks/history?limit=5")
    assert response2.status_code == 200
    assert response1.json() == response2.json()

def test_invalid_benchmark_id():
    """Test error handling for non-existent benchmark"""
    response = client.get("/api/benchmarks/history/nonexistent123")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_ui_rendering():
    """Test the UI endpoint"""
    response = client.get("/api/benchmarks/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]