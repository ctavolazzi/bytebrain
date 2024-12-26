"""Async client for Ollama API with OpenAI compatibility."""
import json
import aiohttp
from typing import AsyncGenerator, Dict, List, Optional, Union

class OllamaError(Exception):
    """Ollama API error."""
    pass

class OllamaClient:
    """Async client for Ollama API with OpenAI compatibility."""

    def __init__(self, host: str = "http://localhost:11434", use_openai_compat: bool = False):
        """Initialize the client with the Ollama API host.

        Args:
            host: Base URL for Ollama API
            use_openai_compat: Whether to use OpenAI-compatible endpoints
        """
        self.host = host.rstrip("/")
        self.use_openai_compat = use_openai_compat

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[Dict, None]:
        """Send a chat request to Ollama API."""
        if self.use_openai_compat:
            url = f"{self.host}/v1/chat/completions"
        else:
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
                        async for line in response.content:
                            line = line.strip()
                            if line:  # Skip empty lines
                                try:
                                    chunk = json.loads(line)
                                    if self.use_openai_compat:
                                        # Transform OpenAI format to Ollama format
                                        if "choices" in chunk and chunk["choices"]:
                                            yield {
                                                "message": {
                                                    "content": chunk["choices"][0]["delta"].get("content", "")
                                                }
                                            }
                                    else:
                                        yield chunk
                                except json.JSONDecodeError as e:
                                    raise OllamaError(f"Failed to parse response: {e}")
                    else:
                        result = await response.json()
                        if self.use_openai_compat:
                            # Transform OpenAI format to Ollama format
                            if "choices" in result and result["choices"]:
                                yield {
                                    "message": {
                                        "content": result["choices"][0]["message"]["content"]
                                    }
                                }
                        else:
                            yield result

        except aiohttp.ClientError as e:
            raise OllamaError(f"Failed to connect to Ollama API: {e}")
        except Exception as e:
            raise OllamaError(f"Unexpected error: {e}")

    async def list_models(self) -> List[str]:
        """Get list of available models."""
        if self.use_openai_compat:
            url = f"{self.host}/v1/models"
        else:
            url = f"{self.host}/api/tags"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if not response.ok:
                        error_text = await response.text()
                        raise OllamaError(f"Ollama API error: {response.status} - {error_text}")

                    result = await response.json()
                    if self.use_openai_compat:
                        return [model["id"] for model in result["data"]]
                    else:
                        return [model["name"] for model in result["models"]]

        except aiohttp.ClientError as e:
            raise OllamaError(f"Failed to connect to Ollama API: {e}")
        except Exception as e:
            raise OllamaError(f"Unexpected error: {e}")
