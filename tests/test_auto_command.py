import pytest
from unittest.mock import MagicMock
import json
import logging
import os

from auto_chat.auto_command import AutoCommandHandler
from auto_chat.session_manager import SessionManager

@pytest.fixture
def mock_bot():
    """Mock bot for testing."""
    bot = MagicMock()
    # Mock streaming response
    bot.chat_stream.return_value = [
        {"role": "assistant", "content": "Response from model"}
    ]
    bot.generate_response.return_value = "Generated context"
    return bot

@pytest.fixture
def test_config():
    """Basic test configuration."""
    return {
        "models": ["llama3.2", "nemotron-mini"],
        "auto_chat": {
            "max_context_length": 1000,
            "max_history_files": 5,
            "context_model": "llama3.2",
            "context_settings": {
                "temperature": 0.7,
                "max_tokens": 500
            },
            "chat_settings": {
                "temperature": 0.7,
                "max_tokens": 500
            },
            "context_prompt_template": "Given the last response: {last_response}\nGenerate a new context.",
            "default_initial_prompt": "Hello! Let's talk."
        }
    }

@pytest.fixture
def handler(mock_bot, test_config, tmp_path):
    """Create a test instance of AutoCommandHandler."""
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(test_config, f)

    handler = AutoCommandHandler(
        bot=mock_bot,
        base_dir=str(tmp_path)
    )
    handler.logger = logging.getLogger(__name__)
    return handler

def test_basic_initialization(handler, mock_bot, test_config):
    """Test that handler initializes with basic components."""
    assert isinstance(handler, AutoCommandHandler)
    assert handler.bot == mock_bot
    assert handler.config == test_config

def test_context_flow(handler, mock_bot):
    """Test the flow of context through the system."""
    # Mock the bot's response for context generation
    mock_bot.generate_response.return_value = "Generated context"

    # Generate initial context
    context = handler.context_manager.generate_context("Hello")

    # Verify context structure
    assert "timestamp" in context
    assert context["scenario"] == "Generated context"
    assert context["initial_prompt"] == "Hello"

    # Verify context is saved
    saved_context = handler.context_manager.get_current_context()
    assert saved_context == context

def test_turn_management(handler, mock_bot):
    """Test turn management and model responses."""
    # Set up mock responses for each model
    mock_bot.chat_stream.side_effect = [
        [{"role": "assistant", "content": "Response from llama3.2"}],  # First call for llama3.2
        [{"role": "assistant", "content": "Response from nemotron-mini"}]   # Second call for nemotron-mini
    ]

    # Mock model_handler to properly format responses
    def mock_get_response(model, prompt, bot):
        response = next(iter(bot.chat_stream()))["content"]
        return {"response": response, "metadata": {}}

    handler.model_handler.get_response = mock_get_response

    # Start with llama3.2
    current_model = handler.turn_manager.get_current()
    assert current_model == "llama3.2"

    # Get response from llama3.2
    response = handler.model_handler.get_response(
        current_model,
        "Hello",
        handler.bot
    )
    assert response["response"] == "Response from llama3.2"

    # Advance turn
    handler.turn_manager.advance()
    current_model = handler.turn_manager.get_current()
    assert current_model == "nemotron-mini"

    # Get response from nemotron-mini
    response = handler.model_handler.get_response(
        current_model,
        "Hello again",
        handler.bot
    )
    assert response["response"] == "Response from nemotron-mini"

    # Advance turn - should go to context manager
    handler.turn_manager.advance()
    assert handler.turn_manager.is_context_manager_turn()
    assert handler.turn_manager.get_current() is None

    # Advance again - should go back to llama3.2
    handler.turn_manager.advance()
    current_model = handler.turn_manager.get_current()
    assert current_model == "llama3.2"

def test_session_setup(tmp_path):
    """Test that SessionManager sets up and provides session info correctly."""
    session_id = "test_session"
    manager = SessionManager(session_id, base_dir=str(tmp_path))

    # Check directories were created
    assert os.path.exists(os.path.join(tmp_path, "sessions", session_id))
    assert os.path.exists(os.path.join(tmp_path, "contexts", session_id))

    # Check session file was initialized
    session_file = os.path.join(tmp_path, "sessions", session_id, "session_data.json")
    assert os.path.exists(session_file)
    with open(session_file) as f:
        data = json.load(f)
        assert data["session_id"] == session_id
        assert "session_start" in data
        assert data["interactions"] == []

    # Check session info is provided correctly
    info = manager.get_session_info()
    assert info["session_id"] == session_id
    assert info["session_dir"] == os.path.join(tmp_path, "sessions", session_id)
    assert info["contexts_dir"] == os.path.join(tmp_path, "contexts", session_id)
    assert info["session_file"] == session_file
    assert info["context_history_file"] == os.path.join(tmp_path, "contexts", session_id, "chat_context_history.json")
    assert info["current_context_file"] == os.path.join(tmp_path, "contexts", session_id, "current_context.txt")

def test_interaction_handling(tmp_path):
    """Test that AutoCommandHandler correctly saves and manages interactions."""
    # Setup
    base_dir = tmp_path / "test_history"
    base_dir.mkdir()
    config = {
        "models": ["llama3.2", "nemotron-mini"],
        "auto_chat": {
            "max_context_length": 1000,
            "max_history_files": 5,
            "context_model": "llama3.2",
            "context_settings": {
                "temperature": 0.7,
                "max_tokens": 500
            },
            "chat_settings": {
                "temperature": 0.7,
                "max_tokens": 500
            },
            "context_prompt_template": "Given the last response: {last_response}\nGenerate a new context.",
            "default_initial_prompt": "Hello! Let's talk."
        }
    }
    config_path = base_dir / "config.json"
    config_path.write_text(json.dumps(config))

    # Create handler
    mock_bot = MagicMock()
    handler = AutoCommandHandler(mock_bot, str(base_dir))
    handler.logger = logging.getLogger(__name__)  # Add logger

    # Mock responses
    mock_response = {"response": "test response", "metadata": {}}
    handler.model_handler.get_response = MagicMock(return_value=mock_response)
    handler.context_manager.generate_context = MagicMock(return_value={
        "timestamp": "2024-01-01T00:00:00",
        "scenario": "test scenario",
        "initial_prompt": "test prompt"
    })

    # Mock the advance method to raise KeyboardInterrupt after first interaction
    original_advance = handler.turn_manager.advance
    def mock_advance():
        original_advance()
        raise KeyboardInterrupt()
    handler.turn_manager.advance = mock_advance

    # Start conversation (will stop after first interaction)
    try:
        handler.start("test prompt")
    except KeyboardInterrupt:
        pass  # Expected to break after first interaction

    # Verify interaction was saved
    assert handler.current_interaction is not None
    assert handler.current_interaction["prompt"] == "test prompt"
    assert handler.current_interaction["model_responses"]["llama3.2"] == mock_response

    # Verify interaction was written to file
    with open(handler.interactions_file, "r") as f:
        saved_interactions = json.load(f)
    assert len(saved_interactions) == 1
    assert saved_interactions[0]["prompt"] == "test prompt"
    assert saved_interactions[0]["model_responses"]["llama3.2"] == mock_response
