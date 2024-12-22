from typing import Generator, Dict
from ollama_client import OllamaClient

class OllamaBot:
    def __init__(self):
        self.client = OllamaClient()
        self.default_model = "llama3.2"

    def get_response(self, prompt: str, model: str = None) -> str:
        """Get a complete response from the model."""
        model = model or self.default_model
        response = self.client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        return response['message']['content']

    def get_streaming_response(self, prompt: str, model: str = None) -> Generator[str, None, None]:
        """Get a streaming response from the model."""
        model = model or self.default_model
        response = self.client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        for chunk in response:
            if chunk and 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']