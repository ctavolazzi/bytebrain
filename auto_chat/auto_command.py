import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from bots.ollama_bot import OllamaBot
from auto_chat.session_manager import SessionManager

logger = logging.getLogger(__name__)

class AutoCommandHandler:
    """Handles automated multi-round conversations between bots."""

    def __init__(self, session_manager: SessionManager):
        """Initialize the handler with a session manager."""
        self.session_manager = session_manager
        self.config = self._load_config()
        self.bot1 = OllamaBot()  # First bot
        self.bot2 = OllamaBot()  # Second bot

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _save_interaction(self, prompt: str, response: str, bot_number: int) -> None:
        """Save an interaction to the session."""
        try:
            model_responses = {
                f"bot{bot_number}": {
                    "response": response,
                    "time_to_first_token": 0,
                    "total_time": 0,
                    "chunk_count": 0,
                    "word_count": len(response.split())
                }
            }
            self.session_manager.save_interaction(prompt, model_responses)
            logger.info(f"Saved interaction from bot{bot_number} to session {self.session_manager.session_id}")
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")

    def start(self, initial_prompt: str, max_iterations: int = 5, stream: bool = True) -> None:
        """Start an automated conversation between two bots.

        Args:
            initial_prompt: The prompt to start the conversation
            max_iterations: Maximum number of back-and-forth exchanges
            stream: Whether to stream the responses
        """
        current_prompt = initial_prompt
        print(f"\nStarting conversation with prompt: {initial_prompt}")

        for i in range(max_iterations):
            print(f"\nRound {i+1}:")

            # Bot 1's turn
            print("\nBot 1: ", end="", flush=True)
            if stream:
                response1 = ""
                for chunk in self.bot1.get_streaming_response(current_prompt):
                    print(chunk, end="", flush=True)
                    response1 += chunk
                print()
            else:
                response1 = self.bot1.get_response(current_prompt)
                print(response1)

            self._save_interaction(current_prompt, response1, 1)
            current_prompt = response1  # Bot 2 will respond to Bot 1's response

            # Bot 2's turn
            print("\nBot 2: ", end="", flush=True)
            if stream:
                response2 = ""
                for chunk in self.bot2.get_streaming_response(current_prompt):
                    print(chunk, end="", flush=True)
                    response2 += chunk
                print()
            else:
                response2 = self.bot2.get_response(current_prompt)
                print(response2)

            self._save_interaction(current_prompt, response2, 2)
            current_prompt = response2  # Next round, Bot 1 will respond to Bot 2's response

        print("\nConversation ended after", max_iterations, "rounds")
