import ollama


from typing import AsyncGenerator, Dict, Optional, Union

class OllamaBot:
    """Bot implementation for interacting with Ollama models."""

    def __init__(self, model: str = "llama3.2", host: Optional[str] = None):
        """Initialize the Ollama bot.

        Args:
            model: The name of the Ollama model to use
            host: Optional host URL for the Ollama API
        """
        self.model = model
        if host:
            ollama.set_host(host)
        self._model_info = None

    async def get_response(self, message: str) -> str:
        """Get a complete response from the model.

        Args:
            message: The input message

        Returns:
            The model's response
        """
        response = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': message}]
        )
        return response['message']['content']

    async def get_streaming_response(self, message: str) -> AsyncGenerator[str, None]:
        """Get a streaming response from the model.

        Args:
            message: The input message

        Yields:
            Response chunks from the model
        """
        stream = ollama.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': message}],
            stream=True
        )

        for chunk in stream:
            if chunk and 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']

    def get_model_version(self) -> str:
        """Get the version of the current model."""
        if not self._model_info:
            try:
                self._model_info = ollama.show(self.model)
            except Exception:
                return "unknown"
        return self._model_info.get('details', {}).get('digest', 'unknown')