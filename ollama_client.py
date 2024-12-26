"""Async client for Ollama API."""
import json
import aiohttp
from typing import AsyncGenerator, Dict, List, Optional

class OllamaError(Exception):
    """Ollama API error."""
    pass

class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize the client with the Ollama API host."""
        self.host = host.rstrip("/")

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[Dict, None]:
        """Send a chat request to Ollama API."""
        url = f"{self.host}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise OllamaError(f"Ollama API error: {response.status} - {error_text}")

                    if stream:
                        # Process streaming response
                        async for line in response.content:
                            line = line.strip()
                            if line:  # Skip empty lines
                                try:
                                    chunk = json.loads(line)
                                    yield chunk
                                except json.JSONDecodeError as e:
                                    raise OllamaError(f"Failed to parse response: {e}")
                    else:
                        # Process non-streaming response
                        result = await response.json()
                        yield result

        except aiohttp.ClientError as e:
            raise OllamaError(f"Failed to connect to Ollama API: {e}")
        except Exception as e:
            raise OllamaError(f"Unexpected error: {e}")

    async def list_models(self) -> List[str]:
        """Get list of available models."""
        url = f"{self.host}/api/tags"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise OllamaError(f"Ollama API error: {response.status} - {error_text}")

                    result = await response.json()
                    return [model["name"] for model in result["models"]]

        except aiohttp.ClientError as e:
            raise OllamaError(f"Failed to connect to Ollama API: {e}")
        except Exception as e:
            raise OllamaError(f"Unexpected error: {e}")
