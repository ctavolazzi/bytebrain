# ByteBrain

A system for automated bot-to-bot conversations using Ollama models.

## Features

- Simple Ollama client for model interactions
- Bot abstraction for easy response handling
- Support for both streaming and non-streaming responses
- Automated multi-round conversations between two bots
- Session management for conversation history

## Installation

1. Make sure you have [Ollama](https://ollama.ai) installed and running
2. Clone this repository
3. Install dependencies:
```bash
pip install poetry
poetry install
```

## Usage

```python
from auto_chat.auto_command import AutoCommandHandler
from auto_chat.session_manager import SessionManager

# Create a new session and handler
handler = AutoCommandHandler(SessionManager.create_new_session())

# Start a conversation with 3 rounds
handler.start(
    initial_prompt="Let's have a conversation about cats. What do you like about them?",
    max_iterations=3,
    stream=True  # Set to False for non-streaming responses
)
```

## Project Structure

- `ollama_client.py` - Simple wrapper for Ollama API
- `bots/ollama_bot.py` - Bot implementation with streaming/non-streaming support
- `auto_chat/` - Core conversation automation
  - `auto_command.py` - Main handler for bot-to-bot conversations
  - `session_manager.py` - Manages conversation history

## Requirements

- Python 3.8+
- Ollama
- Poetry for dependency management
