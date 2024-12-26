"""Tests for the Autogen benchmarking system."""
import pytest
import os
from fastapi.testclient import TestClient
from datetime import datetime

from ag_benchmarking import app
from ag_benchmarking.models.ag_benchmark import (
    LLMConfig,
    AgentConfig,
    ConversationConfig,
    BenchmarkRequest
)

client = TestClient(app)

# Test configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

@pytest.fixture
def basic_conversation_config():
    """Create a basic conversation config with an assistant and user proxy."""
    return ConversationConfig(
        name="basic_test",
        agents=[
            AgentConfig(
                name="assistant",
                type="assistant",
                llm_config=LLMConfig(
                    provider="openai",
                    model="gpt-3.5-turbo",
                    api_key=OPENAI_API_KEY
                ),
                system_message="You are a helpful AI assistant."
            ),
            AgentConfig(
                name="user",
                type="user_proxy",
                system_message="You are a user seeking assistance."
            )
        ],
        initiator="user",
        max_rounds=3,
        description="Basic test conversation"
    )

@pytest.fixture
def multi_agent_config():
    """Create a configuration with multiple agents."""
    return ConversationConfig(
        name="multi_agent_test",
        agents=[
            AgentConfig(
                name="planner",
                type="assistant",
                llm_config=LLMConfig(
                    provider="openai",
                    model="gpt-4",
                    api_key=OPENAI_API_KEY
                ),
                system_message="You are a planning assistant that breaks down tasks."
            ),
            AgentConfig(
                name="coder",
                type="assistant",
                llm_config=LLMConfig(
                    provider="anthropic",
                    model="claude-2",
                    api_key=ANTHROPIC_API_KEY
                ),
                system_message="You are a coding assistant that implements solutions."
            ),
            AgentConfig(
                name="user",
                type="user_proxy",
                system_message="You are a user seeking assistance."
            )
        ],
        initiator="user",
        max_rounds=5,
        description="Multi-agent test conversation"
    )

def test_basic_conversation_benchmark():
    """Test a basic conversation benchmark."""
    config = basic_conversation_config()
    request = BenchmarkRequest(
        prompt="What is the capital of France?",
        configs=[config],
        parallel_processing=False
    )

    response = client.post("/api/benchmarks/run", json=request.dict())
    assert response.status_code == 200

    data = response.json()
    assert data["prompt"] == request.prompt
    assert data["processing_mode"] == "sequential"
    assert len(data["results"]) == 1

    result = data["results"][0]
    assert result["success"] is True
    assert "timing" in result
    assert "system_impact" in result

def test_parallel_multi_agent_benchmark():
    """Test parallel processing with multiple agent configurations."""
    configs = [
        basic_conversation_config(),
        multi_agent_config()
    ]

    request = BenchmarkRequest(
        prompt="Design and implement a Python function to calculate Fibonacci numbers.",
        configs=configs,
        parallel_processing=True
    )

    response = client.post("/api/benchmarks/run", json=request.dict())
    assert response.status_code == 200

    data = response.json()
    assert data["processing_mode"] == "parallel"
    assert len(data["results"]) == 2

def test_error_handling():
    """Test error handling with invalid configurations."""
    # Test with invalid LLM config
    invalid_config = ConversationConfig(
        name="invalid_test",
        agents=[
            AgentConfig(
                name="assistant",
                type="assistant",
                llm_config=LLMConfig(
                    provider="openai",
                    model="gpt-3.5-turbo",
                    # Missing API key
                )
            ),
            AgentConfig(
                name="user",
                type="user_proxy"
            )
        ],
        initiator="user"
    )

    request = BenchmarkRequest(
        prompt="This should fail",
        configs=[invalid_config]
    )

    response = client.post("/api/benchmarks/run", json=request.dict())
    assert response.status_code == 500
    assert "OpenAI API key is required" in response.json()["detail"]

def test_benchmark_history():
    """Test benchmark history endpoints."""
    # First run a benchmark to ensure there's history
    config = basic_conversation_config()
    request = BenchmarkRequest(
        prompt="Create some benchmark history.",
        configs=[config]
    )
    client.post("/api/benchmarks/run", json=request.dict())

    # Test history endpoint
    response = client.get("/api/benchmarks/history")
    assert response.status_code == 200
    history = response.json()
    assert len(history) > 0

    # Test specific benchmark retrieval
    benchmark_id = history[0]["id"]
    response = client.get(f"/api/benchmarks/history/{benchmark_id}")
    assert response.status_code == 200
    assert response.json()["prompt"] == "Create some benchmark history."

def test_streaming_updates():
    """Test streaming updates during benchmark execution."""
    config = basic_conversation_config()
    request = BenchmarkRequest(
        prompt="Test streaming updates.",
        configs=[config]
    )

    # Start streaming connection
    with client.websocket_connect("/api/benchmarks/stream") as websocket:
        # Run benchmark
        client.post("/api/benchmarks/run", json=request.dict())

        # Collect updates
        updates = []
        for _ in range(10):  # Collect up to 10 updates or until completion
            data = websocket.receive_json()
            updates.append(data)
            if data.get("status") == "completed":
                break

        assert len(updates) > 0
        assert any(u["status"] == "starting" for u in updates)
        assert any(u["status"] == "completed" for u in updates)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])