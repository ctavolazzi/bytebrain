"""Comprehensive benchmark script for Ollama models and system performance."""

import time
import psutil
import platform
import json
from datetime import datetime
import GPUtil
from ollama_client import OllamaClient
import os

# Test prompts - we'll move these to a separate file later
TEST_PROMPTS = [
    {
        "name": "pangram",
        "prompt": "Reply with ONLY this sentence, nothing else: The Quick Brown Fox Jumps Over The Lazy Dog"
    },
    {
        "name": "simple_math",
        "prompt": "What is 2 + 2? Reply with just the number."
    },
    {
        "name": "single_word",
        "prompt": "Reply with just the word 'Hello' and nothing else."
    }
]

def get_system_info():
    """Collect detailed system information."""
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        "min_frequency": psutil.cpu_freq().min if psutil.cpu_freq() else None,
        "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
    }

    memory = psutil.virtual_memory()
    memory_info = {
        "total": memory.total / (1024 ** 3),  # GB
        "available": memory.available / (1024 ** 3),  # GB
        "used": memory.used / (1024 ** 3),  # GB
        "percent_used": memory.percent
    }

    # Get GPU information if available
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = [{
            "name": gpu.name,
            "memory_total": gpu.memoryTotal,
            "memory_used": gpu.memoryUsed,
            "memory_free": gpu.memoryFree,
            "temperature": gpu.temperature,
            "load": gpu.load
        } for gpu in gpus]
    except:
        gpu_info = None

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu": cpu_info,
        "memory": memory_info,
        "gpu": gpu_info
    }

def get_performance_metrics():
    """Get current system performance metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
        "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        "memory_percent": psutil.virtual_memory().percent,
        "swap_percent": psutil.swap_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "network": {
            "bytes_sent": psutil.net_io_counters().bytes_sent,
            "bytes_recv": psutil.net_io_counters().bytes_recv
        }
    }

def test_model(client, model_name: str, prompt: str):
    """Test a single model and measure detailed performance metrics."""
    print(f"\nTesting {model_name}...")
    print("-" * 50)

    try:
        # Initial system state
        start_metrics = get_performance_metrics()
        start_time = time.time()
        first_chunk_time = None
        chunk_times = []
        chunk_sizes = []

        # Get streaming response
        response_stream = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        # Collect response and measure timing
        full_response = ""
        last_chunk_time = start_time

        for chunk in response_stream:
            current_time = time.time()

            if not first_chunk_time:
                first_chunk_time = current_time
                print(f"Time to first chunk: {first_chunk_time - start_time:.2f} seconds")

            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                full_response += content

                # Record chunk metrics
                chunk_times.append(current_time - last_chunk_time)
                chunk_sizes.append(len(content.encode('utf-8')))
                last_chunk_time = current_time

        # Final timing and metrics
        end_time = time.time()
        end_metrics = get_performance_metrics()
        total_time = end_time - start_time

        print(f"Total response time: {total_time:.2f} seconds")
        print("\nResponse:")
        print(full_response)
        print("-" * 50)

        # Calculate detailed metrics
        return {
            "model": model_name,
            "timing": {
                "time_to_first_chunk": first_chunk_time - start_time if first_chunk_time else None,
                "total_time": total_time,
                "average_chunk_time": sum(chunk_times) / len(chunk_times) if chunk_times else None,
                "max_chunk_time": max(chunk_times) if chunk_times else None,
                "min_chunk_time": min(chunk_times) if chunk_times else None
            },
            "throughput": {
                "total_chunks": len(chunk_times),
                "total_bytes": sum(chunk_sizes),
                "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else None,
                "bytes_per_second": sum(chunk_sizes) / total_time if total_time > 0 else None
            },
            "system_impact": {
                "cpu_delta": [end - start for start, end in zip(start_metrics["cpu_percent"], end_metrics["cpu_percent"])],
                "memory_delta": end_metrics["memory_percent"] - start_metrics["memory_percent"],
                "start_metrics": start_metrics,
                "end_metrics": end_metrics
            },
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": full_response
        }

    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")
        return {
            "model": model_name,
            "error": str(e),
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt
        }

def run_benchmark_batch(client, models, prompts):
    """Run benchmarks for multiple prompts across all models."""
    all_results = []
    system_info = get_system_info()

    for prompt_info in prompts:
        print(f"\nRunning benchmark for prompt: {prompt_info['name']}")
        print(f"Prompt: {prompt_info['prompt']}")
        print("-" * 50)

        prompt_results = []
        for model in models:
            result = test_model(client, model, prompt_info['prompt'])
            result["prompt_name"] = prompt_info["name"]
            prompt_results.append(result)

        # Save results for this prompt
        save_benchmark_results(
            prompt_results,
            system_info,
            subfolder=prompt_info["name"]
        )
        all_results.extend(prompt_results)

        # Print summary after each prompt
        print(f"\nResults for prompt '{prompt_info['name']}':")
        print_detailed_summary(prompt_results, system_info)
        print("\n" + "="*50 + "\n")

    return all_results

def save_benchmark_results(results, system_info, subfolder=None):
    """Save benchmark results to a JSON file in the benchmark_results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create base directory
    base_dir = "benchmark_results"
    os.makedirs(base_dir, exist_ok=True)

    # Create subfolder if specified
    if subfolder:
        base_dir = os.path.join(base_dir, subfolder)
        os.makedirs(base_dir, exist_ok=True)

    filename = os.path.join(base_dir, f"benchmark_{timestamp}.json")

    data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "results": results
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDetailed results saved to {filename}")

def print_detailed_summary(results, system_info):
    """Print a detailed summary of the benchmark results."""
    print("\nSystem Information:")
    print("-" * 50)
    print(f"Platform: {system_info['platform']}")
    print(f"Processor: {system_info['processor']}")
    print(f"CPU Cores: {system_info['cpu']['physical_cores']} physical, {system_info['cpu']['total_cores']} total")
    print(f"Memory: {system_info['memory']['total']:.1f}GB total, {system_info['memory']['available']:.1f}GB available")
    if system_info['gpu']:
        for i, gpu in enumerate(system_info['gpu']):
            print(f"GPU {i+1}: {gpu['name']}, {gpu['memory_total']}MB memory")

    print("\nBenchmark Results:")
    print("-" * 50)
    for result in results:
        if result["success"]:
            print(f"\n{result['model']}:")
            print(f"  Response Timing:")
            print(f"    First chunk: {result['timing']['time_to_first_chunk']:.2f}s")
            print(f"    Total time: {result['timing']['total_time']:.2f}s")
            print(f"    Avg chunk time: {result['timing']['average_chunk_time']:.3f}s")

            print(f"  Throughput:")
            print(f"    Total chunks: {result['throughput']['total_chunks']}")
            print(f"    Total bytes: {result['throughput']['total_bytes']}")
            print(f"    Bytes/second: {result['throughput']['bytes_per_second']:.2f}")

            print(f"  System Impact:")
            print(f"    CPU delta: {max(result['system_impact']['cpu_delta']):.1f}%")
            print(f"    Memory delta: {result['system_impact']['memory_delta']:.1f}%")
        else:
            print(f"\n{result['model']}: Failed - {result['error']}")
        print("-" * 25)

def main():
    # Initialize client
    client = OllamaClient()

    # List of models to test
    models = ["wizardlm2", "nemotron-mini", "llama3.2"]

    print("Starting comprehensive benchmark batch...")
    print(f"Models to test: {', '.join(models)}")
    print(f"Number of prompts: {len(TEST_PROMPTS)}")

    # Run all benchmarks
    all_results = run_benchmark_batch(client, models, TEST_PROMPTS)

    # Save combined results
    system_info = get_system_info()
    save_benchmark_results(all_results, system_info, subfolder="combined")

if __name__ == "__main__":
    main()