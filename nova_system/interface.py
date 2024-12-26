import asyncio
import logging
from typing import AsyncGenerator, Dict, Optional, Union

from .core import NovaSystemCore
from .agents import AgentOrchestrator

logger = logging.getLogger(__name__)

class NovaSystem:
    """Main interface for the NovaSystem Autogen Ollama Local LLM Bot."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the NovaSystem interface.

        Args:
            config_path: Path to the configuration file
        """
        self.core = NovaSystemCore(config_path)
        self.orchestrator = AgentOrchestrator()
        logger.info("NovaSystem initialized")

    async def process_message(
        self,
        message: str,
        stream: bool = True
    ) -> Union[AsyncGenerator[str, None], str]:
        """Process a user message through the system.

        Args:
            message: The user's input message
            stream: Whether to stream the response

        Returns:
            Either a string response or an async generator of response chunks
        """
        try:
            # Process through agent orchestration
            orchestrator_response = await self.orchestrator.process_turn(message)

            # Process through core system
            if stream:
                async def response_generator():
                    async for chunk in self.core.process_message(orchestrator_response, stream=True):
                        yield chunk
                return response_generator()
            else:
                return await self.core.process_message(orchestrator_response, stream=False)

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            if stream:
                async def error_generator():
                    yield error_msg
                return error_generator()
            return error_msg

    def get_session_info(self) -> Dict:
        """Get information about the current session."""
        return {
            "session_id": self.core.session_id,
            "conversation_summary": self.core.get_conversation_summary(),
            "agent_steps": self.orchestrator.get_agent_steps()
        }

    async def close(self) -> None:
        """Clean up resources and close the session."""
        # Add any cleanup code here
        logger.info("NovaSystem session closed")