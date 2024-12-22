from typing import Generator, Dict, Union
from ollama import Client

class OllamaClient:
    def __init__(self, host: str = 'http://localhost:11434'):
        self.client = Client(host=host)

    def chat(self, model: str, messages: list, stream: bool = True) -> Union[Generator[Dict, None, None], Dict]:
        return self.client.chat(
            model=model,
            messages=messages,
            stream=stream
        )
