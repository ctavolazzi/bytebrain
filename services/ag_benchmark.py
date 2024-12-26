"""Benchmark service for Autogen conversations."""
import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..services.ag_client import AutogenClient, AutogenError, ConversationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkService:
    """Service for benchmarking Autogen conversations."""

    def __init__(self):
        """Initialize the benchmark service."""
        self.autogen_client = AutogenClient()

    def get_system_info(self) -> Dict:
        """Get current system information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "platform": psutil.sys_platform,
                "python_version": psutil.python_version(),
                "cpu": {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "total_cores": psutil.cpu_count(logical=True),
                    "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                },
                "memory": {
                    "total": memory.total / (1024 ** 3),  # Convert to GB
                    "available": memory.available / (1024 ** 3),
                    "used": memory.used / (1024 ** 3),
                    "percent_used": memory.percent,
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}

    async def benchmark_conversation(
        self,
        config: ConversationConfig,
        prompt: str,
        update_queue: Optional[asyncio.Queue] = None
    ) -> Dict:
        """Run a benchmark for a single conversation configuration."""
        logger.info(f"Starting benchmark for conversation with {len(config.agents)} agents")

        try:
            # Record initial system metrics
            initial_cpu = psutil.cpu_percent(interval=None, percpu=True)
            initial_memory = psutil.virtual_memory().used / (1024 ** 3)

            # Start timing
            start_time = datetime.utcnow()
            start_monotonic = time.monotonic()

            # Send starting update
            if update_queue:
                await update_queue.put({
                    "status": "starting",
                    "message": f"Starting benchmark with {len(config.agents)} agents"
                })

            # Run the conversation
            result = await self.autogen_client.run_conversation(
                config=config,
                prompt=prompt,
                update_queue=update_queue
            )

            # Record final system metrics
            final_cpu = psutil.cpu_percent(interval=None, percpu=True)
            final_memory = psutil.virtual_memory().used / (1024 ** 3)

            # Calculate total time
            total_time = time.monotonic() - start_monotonic

            # Add system metrics to result
            result.update({
                "system_impact": {
                    "cpu_delta": [f - i for f, i in zip(final_cpu, initial_cpu)],
                    "memory_delta": final_memory - initial_memory
                },
                "timestamp": start_time.isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def run_benchmark(
        self,
        prompt: str,
        configs: List[ConversationConfig],
        parallel_processing: bool = False,
        parameters: Optional[Dict] = None
    ) -> Dict:
        """Run benchmarks for multiple conversation configurations."""
        logger.info(f"Running {'parallel' if parallel_processing else 'sequential'} benchmark with {len(configs)} configurations")

        # Create a queue for streaming updates
        update_queue = asyncio.Queue()

        try:
            # Get initial system info
            system_info = self.get_system_info()

            # Run benchmarks
            if parallel_processing:
                # Run all conversations in parallel
                tasks = [
                    self.benchmark_conversation(
                        config=config,
                        prompt=prompt,
                        update_queue=update_queue
                    )
                    for config in configs
                ]
                results = await asyncio.gather(*tasks)
            else:
                # Run conversations sequentially
                results = []
                for config in configs:
                    result = await self.benchmark_conversation(
                        config=config,
                        prompt=prompt,
                        update_queue=update_queue
                    )
                    results.append(result)

            # Compile final results
            return {
                "prompt": prompt,
                "processing_mode": "parallel" if parallel_processing else "sequential",
                "parameters": parameters,
                "system_info": system_info,
                "results": results,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Benchmark run failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        finally:
            # Reset the Autogen client
            self.autogen_client.reset()