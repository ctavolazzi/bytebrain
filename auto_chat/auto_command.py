import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from .session_manager import SessionManager
from bots.ollama_bot import OllamaBot
from nova_system.core import NovaSystemCore
from nova_system.agents import AgentOrchestrator

class AutoCommandHandler:
    """Handles automated chat commands and interactions."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the command handler.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.session_manager = SessionManager()
        self.bot = OllamaBot(model=self.config.get("model", "llama3.2"))
        self.nova_core = NovaSystemCore()
        self.agent_orchestrator = AgentOrchestrator()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    async def process_message(self, message: str, stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """Process a user message and get response.

        Args:
            message: The user's message
            stream: Whether to stream the response

        Returns:
            The bot's response or a response stream
        """
        # Log the interaction start
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "type": "stream" if stream else "complete"
        }

        try:
            # Process through Nova System
            nova_context = await self.nova_core.process_message(message)

            # Get response through agent orchestration
            agent_result = await self.agent_orchestrator.process_turn(message)

            # Get bot response
            if stream:
                async def response_stream():
                    async for chunk in self.bot.get_streaming_response(message):
                        yield chunk
                response = response_stream()
            else:
                response = await self.bot.get_response(message)

            # Update interaction with response
            interaction.update({
                "status": "success",
                "nova_context": nova_context,
                "agent_result": agent_result,
                "response": response if not stream else "streaming"
            })

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            interaction.update({
                "status": "error",
                "error": error_msg
            })
            response = error_msg

        # Save interaction
        self.session_manager.add_interaction(interaction)

        return response

    def save_interaction(self, interaction: Dict):
        """Save an interaction to the current session.

        Args:
            interaction: The interaction to save
        """
        self.session_manager.add_interaction(interaction)

    def get_session_info(self) -> Dict:
        """Get information about the current session.

        Returns:
            Dictionary containing session information
        """
        return {
            "session": self.session_manager.get_session_summary(),
            "nova_system": self.nova_core.get_conversation_summary(),
            "agent_states": self.agent_orchestrator.get_agent_states(),
            "model_version": self.bot.get_model_version()
        }

    def close(self):
        """Clean up resources and close the session."""
        self.session_manager.close_session()
