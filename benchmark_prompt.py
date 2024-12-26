"""Simple interface for benchmarking Ollama models with user input."""

from benchmark_ollama import (
    OllamaClient,
    test_model,
    print_detailed_summary,
    save_benchmark_results,
    get_system_info
)

def main():
    print("\nOllama Model Benchmark")
    print("=" * 50)
    print("Available models: wizardlm2, nemotron-mini, llama3.2")
    print("Enter your prompt below (Ctrl+C to exit)")
    print("-" * 50)

    try:
        # Get user input
        prompt = input("\nYour prompt: ").strip()
        if not prompt:
            print("Prompt cannot be empty!")
            return

        # Initialize
        client = OllamaClient()
        models = ["wizardlm2", "nemotron-mini", "llama3.2"]
        system_info = get_system_info()

        # Run benchmarks
        print("\nRunning benchmarks...")
        results = []
        for model in models:
            result = test_model(client, model, prompt)
            results.append(result)

        # Save results
        save_benchmark_results(
            results,
            system_info,
            subfolder="user_prompts"
        )

        # Print summary
        print_detailed_summary(results, system_info)

    except KeyboardInterrupt:
        print("\nBenchmark cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()