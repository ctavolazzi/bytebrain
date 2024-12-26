import asyncio
import platform
import psutil
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional

import GPUtil
from app.models.benchmark import SystemInfo, BenchmarkResult, BenchmarkResponse
from ollama_client import OllamaClient, OllamaError

class BenchmarkService:
    def __init__(self):
        """Initialize the benchmark service with an Ollama client."""
        self.ollama = OllamaClient()

    @staticmethod
    async def get_system_info() -> SystemInfo:
        """Gather system information for benchmarking context."""
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "min_frequency": psutil.cpu_freq().min if psutil.cpu_freq() else None,
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        }

        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total / (1024 ** 3),  # Convert to GB
            "available": memory.available / (1024 ** 3),
            "used": memory.used / (1024 ** 3),
            "percent_used": memory.percent
        }

        # Get GPU information if available
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "temperature": gpu.temperature
                }
                for gpu in gpus
            ]
        except:
            gpu_info = []

        return SystemInfo(
            platform=platform.platform(),
            processor=platform.processor(),
            python_version=sys.version.split()[0],
            cpu=cpu_info,
            memory=memory_info,
            gpu=gpu_info if gpu_info else None
        )

    async def benchmark_model(self, model: str, prompt: str, parameters: Optional[Dict] = None, update_queue: Optional[asyncio.Queue] = None) -> BenchmarkResult:
        """Benchmark a single model with the given prompt."""
        # Use monotonic time for accurate duration measurements
        start_monotonic = time.monotonic()
        start_time = datetime.utcnow()
        first_token_time = None
        chunks_received = 0
        total_bytes = 0
        response_text = ""

        # Record initial system metrics - do this quickly without interval
        initial_cpu = psutil.cpu_percent(interval=None, percpu=True)
        initial_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

        try:
            # Format message for Ollama
            messages = [{"role": "user", "content": prompt}]

            # Send initial status
            if update_queue:
                await update_queue.put({
                    "model": model,
                    "status": "starting",
                    "time_elapsed": 0
                })

            # Get the streaming response
            async for chunk in self.ollama.chat(model=model, messages=messages, stream=True):
                current_time = time.monotonic() - start_monotonic

                if chunks_received == 0:
                    # Calculate time to first token using monotonic time
                    first_token_time = current_time
                    if update_queue:
                        await update_queue.put({
                            "model": model,
                            "status": "first_token",
                            "time_elapsed": current_time,
                            "time_to_first_token": first_token_time
                        })

                chunk_content = chunk.get("message", {}).get("content", "")
                if chunk_content:  # Only count non-empty chunks
                    chunk_bytes = len(chunk_content.encode())
                    total_bytes += chunk_bytes
                    chunks_received += 1
                    response_text += chunk_content

                    # Send every chunk immediately
                    if update_queue:
                        await update_queue.put({
                            "model": model,
                            "status": "generating",
                            "time_elapsed": current_time,
                            "time_to_first_token": first_token_time,
                            "tokens_generated": chunks_received,
                            "tokens_per_second": chunks_received / current_time if current_time > 0 else 0,
                            "chunk": chunk_content,
                            "response_so_far": response_text
                        })

            # Record final system metrics - do this quickly without interval
            final_cpu = psutil.cpu_percent(interval=None, percpu=True)
            final_memory = psutil.virtual_memory().used / (1024 ** 3)

            # Calculate total time using monotonic time
            total_time = time.monotonic() - start_monotonic

            # If we got no chunks, consider it a failure
            if chunks_received == 0:
                raise OllamaError("No response received from model")

            result = BenchmarkResult(
                model=model,
                timing={
                    "time_to_first_token": first_token_time,
                    "total_time": total_time
                },
                throughput={
                    "total_chunks": chunks_received,
                    "total_bytes": total_bytes,
                    "average_chunk_size": total_bytes / chunks_received if chunks_received > 0 else 0,
                    "bytes_per_second": total_bytes / total_time if total_time > 0 else 0
                },
                system_impact={
                    "cpu_delta": [f - i for f, i in zip(final_cpu, initial_cpu)],
                    "memory_delta": final_memory - initial_memory
                },
                success=True,
                timestamp=start_time,
                response=response_text
            )

            # Send completion update
            if update_queue:
                await update_queue.put({
                    "model": model,
                    "status": "completed",
                    "time_elapsed": total_time,
                    "time_to_first_token": first_token_time,
                    "tokens_generated": chunks_received,
                    "tokens_per_second": chunks_received / total_time if total_time > 0 else 0
                })

            return result

        except Exception as e:
            # Calculate total time even for errors
            total_time = time.monotonic() - start_monotonic

            # Send error update
            if update_queue:
                await update_queue.put({
                    "model": model,
                    "status": "error",
                    "time_elapsed": total_time,
                    "error": str(e)
                })

            return BenchmarkResult(
                model=model,
                timing={
                    "error": str(e),
                    "total_time": total_time
                },
                throughput={
                    "total_chunks": chunks_received,
                    "total_bytes": total_bytes,
                    "average_chunk_size": total_bytes / chunks_received if chunks_received > 0 else 0,
                    "bytes_per_second": total_bytes / total_time if total_time > 0 else 0
                },
                system_impact={},
                success=False,
                timestamp=start_time,
                response=response_text if response_text else None
            )

    async def run_benchmark(self, prompt: str, models: List[str], parameters: Optional[Dict] = None, update_queue: Optional[asyncio.Queue] = None) -> BenchmarkResponse:
        """Run benchmarks for multiple models in parallel."""
        # Gather system information
        logger.info("���� Gathering system information...")
        system_info = await BenchmarkService.get_system_info()
        logger.info("✅ System information gathered successfully")

        # Run benchmarks for all models concurrently
        tasks = [
            self.benchmark_model(model, prompt, parameters, update_queue)
            for model in models
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, converting exceptions to failed benchmark results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(BenchmarkResult(
                    model=models[i],
                    timing={"error": str(result)},
                    throughput={},
                    system_impact={},
                    success=False,
                    timestamp=datetime.utcnow(),
                    response=None
                ))
            else:
                processed_results.append(result)

        # Create the response with the prompt included
        return BenchmarkResponse(
            timestamp=datetime.utcnow(),
            system_info=system_info,
            prompt=prompt,  # Ensure prompt is included in response
            results=processed_results
        )