from typing import Dict, Any, Optional
from bots.ollama_bot import OllamaBot

class ModelHandler:
    """Handles interactions with different models."""

    def __init__(self, bot: Optional[OllamaBot] = None):
        """Initialize the model handler.

        Args:
            bot: Optional OllamaBot instance. If not provided, creates a new one.
        """
        self.bot = bot or OllamaBot()

    def get_response(self, model: str, prompt: str, bot: Optional[OllamaBot] = None) -> Dict[str, Any]:
        """Get a response from a specific model.

        Args:
            model: Name of the model to use
            prompt: Input prompt
            bot: Optional bot instance to use instead of the default one

        Returns:
            Dict containing response and metadata
        """
        try:
            bot = bot or self.bot
            response = bot.generate_response(prompt, model=model)

            if isinstance(response, dict):
                return response

            # If response is just a string (no benchmarking), wrap it
            return {
                'response': response,
                'metadata': {}
            }

        except Exception as e:
            print(f"Error getting response from {model}: {e}")
            return {
                'response': f"Error: {str(e)}",
                'metadata': {'error': str(e)}
            }