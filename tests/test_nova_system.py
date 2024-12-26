import pytest
from nova_system.core import NovaSystemCore
import json
import os
from datetime import datetime
import tempfile

@pytest.fixture
def nova_system():
    """Create a NovaSystem instance for testing."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "model": "wizardlm2",
            "temperature": 0.7,
            "max_tokens": 512,
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }, f)
        config_path = f.name

    system = NovaSystemCore(config_path=config_path)
    yield system

    # Cleanup
    system.cleanup()
    if os.path.exists(config_path):
        os.remove(config_path)

@pytest.mark.asyncio
async def test_process_message_success(nova_system):
    """Test successful message processing with Ollama."""
    # Process a test message
    result = await nova_system.process_message("Hello, how are you?")

    # Verify the structure of the response
    assert isinstance(result, dict)
    assert "response" in result
    assert "success" in result
    assert "processing_time_seconds" in result
    assert "user_interaction" in result
    assert "assistant_interaction" in result

    # Verify success status
    assert result["success"] is True

    # Verify response is not empty
    assert result["response"].strip() != ""

    # Verify interactions were logged
    assert result["user_interaction"]["role"] == "user"
    assert result["assistant_interaction"]["role"] == "assistant"

    # Verify metadata
    assert "model_details" in result["assistant_interaction"]["metadata"]
    assert "processing_time_seconds" in result["assistant_interaction"]["metadata"]

@pytest.mark.asyncio
async def test_process_message_with_chain_steps(nova_system):
    """Test message processing with chain steps."""
    chain_steps = [
        {
            "agent_name": "PlannerAgent",
            "input": "Plan response",
            "output": "Planned steps",
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_ms": 100
        }
    ]

    result = await nova_system.process_message(
        "Test message with chain steps",
        chain_steps=chain_steps
    )

    # Verify chain steps were included
    assert "chain_steps" in result["assistant_interaction"]
    assert len(result["assistant_interaction"]["chain_steps"]) == 1
    assert result["assistant_interaction"]["chain_steps"][0]["agent_name"] == "PlannerAgent"

@pytest.mark.asyncio
async def test_process_message_error_handling(nova_system, monkeypatch):
    """Test error handling during message processing."""
    # Mock Ollama client to raise an exception
    def mock_chat(*args, **kwargs):
        raise Exception("Test error")

    monkeypatch.setattr("ollama_client.OllamaClient.chat", mock_chat)

    result = await nova_system.process_message("Test error handling")

    # Verify error response
    assert result["success"] is False
    assert "error" in result
    assert "error_interaction" in result
    assert result["error_interaction"]["role"] == "system"
    assert "error_type" in result["error_interaction"]["metadata"]

def test_config_loading(nova_system):
    """Test configuration loading."""
    assert "model" in nova_system.config
    assert "temperature" in nova_system.config
    assert "max_tokens" in nova_system.config
    assert "logging" in nova_system.config

def test_cleanup(nova_system):
    """Test cleanup functionality."""
    # Create some test files
    test_file = "test_config.json"
    with open(test_file, 'w') as f:
        json.dump({}, f)

    # Initialize a new system with the test file
    system = NovaSystemCore(config_path=test_file)

    # Clean up
    system.cleanup()

    # Verify cleanup
    assert not os.path.exists(test_file)
    assert len(system.logger.handlers) == 0