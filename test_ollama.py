"""Benchmark script for testing all installed Ollama models."""

import time
from datetime import datetime
from ollama_client import OllamaClient

def test_model(client, model_name: str, prompt: str):
    """Test a single model and measure its response times."""
    print(f"\nTesting {model_name}...")
    print("-" * 50)

    try:
        # Start timing
        start_time = time.time()
        first_chunk_time = None

        # Get streaming response
        response_stream = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        # Collect response and measure timing
        full_response = ""
        for chunk in response_stream:
            if not first_chunk_time:
                first_chunk_time = time.time()
                print(f"Time to first chunk: {first_chunk_time - start_time:.2f} seconds")

            if "message" in chunk and "content" in chunk["message"]:
                full_response += chunk["message"]["content"]

        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time

        print(f"Total response time: {total_time:.2f} seconds")
        print("\nResponse:")
        print(full_response)
        print("-" * 50)

        return {
            "model": model_name,
            "time_to_first_chunk": first_chunk_time - start_time if first_chunk_time else None,
            "total_time": total_time,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")
        return {
            "model": model_name,
            "error": str(e),
            "success": False,
            "timestamp": datetime.now().isoformat()
        }

def main():
    # Initialize client
    client = OllamaClient()

    # List of models to test
    models = ["wizardlm2", "nemotron-mini", "llama3.2"]

    # Test prompt
    prompt = "Reply with ONLY this sentence, nothing else: The Quick Brown Fox Jumps Over The Lazy Dog"

    print("Starting model benchmark...")
    print(f"Test prompt: {prompt}")

    # Test each model
    results = []
    for model in models:
        result = test_model(client, model, prompt)
        results.append(result)

    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 50)
    for result in results:
        if result["success"]:
            print(f"{result['model']}:")
            print(f"  Time to first chunk: {result['time_to_first_chunk']:.2f}s")
            print(f"  Total time: {result['total_time']:.2f}s")
        else:
            print(f"{result['model']}: Failed - {result['error']}")
        print("-" * 25)

if __name__ == "__main__":
    main()