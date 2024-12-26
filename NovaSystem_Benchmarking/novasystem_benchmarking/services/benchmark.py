import asyncio
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import psutil
import platform
import sys

from ..models.benchmark import SystemInfo, BenchmarkResponse, BenchmarkResult
from ..services.ollama_client import OllamaClient
from ..services.stream import send_update  # Import the stream update function

# Set up logging
logger = logging.getLogger(__name__)

class BenchmarkService:
    def __init__(self):
        """Initialize the benchmark service."""
        self.ollama_client = OllamaClient()

    @staticmethod
    def get_system_info() -> SystemInfo:
        """Gather system information for benchmarking context."""
        logger.info("Gathering system information...")

        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "min_frequency": psutil.cpu_freq().min if psutil.cpu_freq() else None,
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        }
        logger.debug(f"CPU Info: {cpu_info}")

        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total / (1024 ** 3),  # Convert to GB
            "available": memory.available / (1024 ** 3),
            "used": memory.used / (1024 ** 3),
            "percent_used": memory.percent
        }
        logger.debug(f"Memory Info: {memory_info}")

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
            logger.debug(f"GPU Info: {gpu_info}")
        except:
            logger.warning("No GPU information available")
            gpu_info = None

        system_info = SystemInfo(
            platform=platform.platform(),
            processor=platform.processor(),
            python_version=sys.version,
            cpu=cpu_info,
            memory=memory_info,
            gpu=gpu_info if gpu_info else None
        )
        logger.info("System information gathered successfully")
        return system_info

    async def benchmark_model(self, model: str, prompt: str, parameters: Optional[Dict] = None) -> BenchmarkResult:
        """Benchmark a single model with the given prompt."""
        logger.info(f"Starting benchmark for model: {model}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        if parameters:
            logger.info(f"Parameters: {parameters}")

        # Use monotonic time for accurate duration measurements
        start_monotonic = time.monotonic()
        start_time = datetime.utcnow()
        first_token_time = None
        chunks_received = 0
        total_bytes = 0
        response_text = ""

        # Record initial system metrics
        initial_cpu = psutil.cpu_percent(interval=None, percpu=True)
        initial_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB
        logger.info(f"Initial CPU usage: {initial_cpu}")
        logger.info(f"Initial memory usage: {initial_memory:.2f} GB")

        try:
            # Format message for Ollama
            messages = [{"role": "user", "content": prompt}]
            logger.info(f"Sending request to model {model}")

            # Send initial status
            await send_update({
                "model": model,
                "status": "starting",
                "message": f"Starting benchmark for {model}..."
            })

            # Get the streaming response
            async for chunk in self.ollama_client.chat(model=model, messages=messages, stream=True):
                if chunks_received == 0:
                    # Calculate time to first token using monotonic time
                    first_token_time = time.monotonic() - start_monotonic
                    logger.info(f"Time to first token: {first_token_time:.3f} seconds")
                    await send_update({
                        "model": model,
                        "status": "first_token",
                        "time": first_token_time,
                        "message": f"First token received after {first_token_time:.2f}s"
                    })

                chunk_content = chunk.get("message", {}).get("content", "")
                if chunk_content:  # Only count non-empty chunks
                    chunk_bytes = len(chunk_content.encode())
                    total_bytes += chunk_bytes
                    chunks_received += 1
                    response_text += chunk_content

                    # Send chunk update
                    await send_update({
                        "model": model,
                        "status": "generating",
                        "chunk": chunk_content,
                        "chunks_received": chunks_received,
                        "total_bytes": total_bytes,
                        "time_elapsed": time.monotonic() - start_monotonic
                    })

            # Send completion status
            await send_update({
                "model": model,
                "status": "completed",
                "message": f"Benchmark completed for {model}",
                "total_time": time.monotonic() - start_monotonic,
                "total_chunks": chunks_received,
                "total_bytes": total_bytes
            })

            # Record final system metrics
            final_cpu = psutil.cpu_percent(interval=None, percpu=True)
            final_memory = psutil.virtual_memory().used / (1024 ** 3)
            logger.info(f"Final CPU usage: {final_cpu}")
            logger.info(f"Final memory usage: {final_memory:.2f} GB")

            # Calculate total time using monotonic time
            total_time = time.monotonic() - start_monotonic
            logger.info(f"Benchmark completed for {model}")
            logger.info(f"Total time: {total_time:.3f} seconds")
            logger.info(f"Chunks received: {chunks_received}")
            logger.info(f"Total bytes: {total_bytes}")

            # If we got no chunks, consider it a failure
            if chunks_received == 0:
                raise OllamaError("No response received from model")

            return BenchmarkResult(
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
        except Exception as e:
            logger.error(f"Error benchmarking {model}: {str(e)}")
            # Calculate total time even for errors
            total_time = time.monotonic() - start_monotonic

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

    async def run_benchmark(
        self,
        prompt: str,
        models: List[str],
        parameters: Optional[Dict] = None,
        parallel_processing: bool = True
    ) -> BenchmarkResponse:
        """Run benchmarks on the specified models."""
        start_time = time.time()

        logger.info("\n" + "="*80)
        logger.info("🚀 BENCHMARK RUN INITIATED")
        logger.info("="*80)
        logger.info("📋 Run Configuration:")
        logger.info(f"🔄 Processing Mode: {'⚡️ Parallel' if parallel_processing else '📝 Sequential'}")
        logger.info(f"📝 Prompt: {prompt}")
        logger.info(f"🤖 Models: {', '.join(models)}")
        if parameters:
            logger.info(f"⚙️  Parameters: {parameters}")
        logger.info("="*80)

        # Gather system information
        logger.info("📊 Gathering system information...")
        system_info = BenchmarkService.get_system_info()
        logger.info("✅ System information gathered successfully")

        # Run benchmarks
        benchmark_start = time.time()
        if parallel_processing:
            logger.info("⚡️ Starting parallel benchmark run")
            tasks = [
                self.benchmark_model(model, prompt, parameters)
                for model in models
            ]
            results = await asyncio.gather(*tasks)
        else:
            logger.info("📝 Starting sequential benchmark run")
            results = []
            for model in models:
                logger.info(f"▶️  Starting benchmark for {model}")
                result = await self.benchmark_model(model, prompt, parameters)
                results.append(result)
                logger.info(f"✅ Completed benchmark for {model}")

        benchmark_duration = time.time() - benchmark_start
        total_duration = time.time() - start_time

        # Create response
        response = BenchmarkResponse(
            timestamp=datetime.utcnow(),
            system_info=system_info,
            prompt=prompt,
            processing_mode="parallel" if parallel_processing else "sequential",
            results=results
        )

        # Display timing summary
        logger.info("\n" + "="*80)
        logger.info("⏱️  TIMING SUMMARY")
        logger.info("="*80)
        logger.info(f"🔄 Processing Mode: {'⚡️ Parallel' if parallel_processing else '📝 Sequential'}")
        logger.info(f"⏱️  Total Run Time: {total_duration:.2f} seconds")
        logger.info(f"⚡️ Pure Benchmark Time: {benchmark_duration:.2f} seconds")
        logger.info(f"📊 Overhead Time: {(total_duration - benchmark_duration):.2f} seconds")

        # Model-specific timing summary
        logger.info("\n🤖 Per-Model Results:")
        for result in results:
            if result.success:
                logger.info(f"  • {result.model}:")
                ttft = result.timing.get('time_to_first_token', 'N/A')
                gen_time = result.timing.get('generation_time', 'N/A')
                tps = result.throughput.get('tokens_per_second', 'N/A')

                logger.info(f"    - Time to First Token: {ttft if isinstance(ttft, str) else f'{ttft:.2f}'}s")
                logger.info(f"    - Generation Time: {gen_time if isinstance(gen_time, str) else f'{gen_time:.2f}'}s")
                logger.info(f"    - Tokens/Second: {tps if isinstance(tps, str) else f'{tps:.2f}'}")
            else:
                logger.info(f"  • {result.model}: ❌ Failed")

        logger.info("="*80)
        logger.info("✅ BENCHMARK RUN COMPLETED")
        logger.info("="*80 + "\n")

        return response