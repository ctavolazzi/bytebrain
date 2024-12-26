"""Example usage of NovaSystem Benchmarking."""
from novasystem_benchmarking import OllamaClient
import asyncio

async def main():
    """Run a simple benchmark example."""
    # Initialize the client
    client = OllamaClient()

    # List available models
    models = await client.list_models()
    print("Available models:", models)

    # Run a chat completion
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    async for response in client.chat("llama2", messages):
        if "response" in response:
            print(response["response"], end="", flush=True)
    print()  # New line at end

if __name__ == "__main__":
    asyncio.run(main())