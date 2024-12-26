"""Core benchmarking functionality for AutoGen agents."""
import os
import time
import psutil
import platform
import autogen
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models.ag_benchmark import (
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResult,
    SystemInfo,
    SystemMetrics,
    TimingMetrics,
    ConversationMetrics
)

class BenchmarkRunner:
    """Runner class for executing AutoGen agent benchmarks."""

    def __init__(self):
        """Initialize the benchmark runner."""
        self.process = psutil.Process()

    def _get_system_info(self) -> SystemInfo:
        """Get current system information."""
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

        return SystemInfo(
            platform=platform.platform(),
            python_version=platform.python_version(),
            cpu=cpu_info,
            memory=memory_info,
            timestamp=datetime.now()
        )

    def _get_system_metrics_start(self) -> Dict[str, Any]:
        """Get system metrics at the start of a benchmark."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1, percpu=True),
            "memory_percent": psutil.virtual_memory().percent,
            "swap_percent": psutil.swap_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }

    def _calculate_system_impact(
        self,
        start_metrics: Dict[str, Any],
        end_metrics: Dict[str, Any]
    ) -> SystemMetrics:
        """Calculate system impact during benchmark."""
        cpu_delta = [end - start for end, start in
                    zip(end_metrics["cpu_percent"], start_metrics["cpu_percent"])]
        memory_delta = end_metrics["memory_percent"] - start_metrics["memory_percent"]

        return SystemMetrics(
            cpu_delta=cpu_delta,
            memory_delta=memory_delta
        )

    def _create_agents(self, config: Dict[str, Any]) -> List[autogen.Agent]:
        """Create AutoGen agents from configuration."""
        agents = []
        for agent_config in config.agents:
            if agent_config.type == "assistant":
                agent = autogen.AssistantAgent(
                    name=agent_config.name,
                    llm_config=agent_config.llm_config.dict(),
                    system_message=agent_config.system_message
                )
            elif agent_config.type == "user_proxy":
                agent = autogen.UserProxyAgent(
                    name=agent_config.name,
                    system_message=agent_config.system_message,
                    human_input_mode="NEVER",
                    code_execution_config={"use_docker": False}  # Disable Docker
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_config.type}")
            agents.append(agent)
        return agents

    def _run_single_benchmark(
        self,
        prompt: str,
        config: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """Run a single benchmark with the given configuration."""
        try:
            # Create agents
            agents = self._create_agents(config)
            initiator = next(a for a in agents if a.name == config.initiator)

            # Start metrics collection
            start_time = time.time()
            start_metrics = self._get_system_metrics_start()

            # Run the conversation
            initiator.initiate_chat(
                agents[0] if agents[0] != initiator else agents[1],
                message=prompt,
                max_turns=config.max_rounds
            )

            # End metrics collection
            end_time = time.time()
            end_metrics = self._get_system_metrics_start()

            # Calculate metrics
            total_time = end_time - start_time
            system_impact = self._calculate_system_impact(start_metrics, end_metrics)

            return BenchmarkResult(
                config_name=config.name,
                success=True,
                timing=TimingMetrics(total_time=total_time),
                metrics=None,  # We'll add conversation metrics later if needed
                system_impact=system_impact,
                timestamp=datetime.now()
            )

        except Exception as e:
            return BenchmarkResult(
                config_name=config.name,
                success=False,
                timing=TimingMetrics(total_time=0, error=str(e)),
                timestamp=datetime.now(),
                error=str(e)
            )

    def run_benchmark(self, request: BenchmarkRequest) -> BenchmarkResponse:
        """Run benchmarks according to the request configuration."""
        system_info = self._get_system_info()
        results = []

        if request.parallel_processing:
            # TODO: Implement parallel processing using asyncio
            pass
        else:
            # Sequential processing
            for config in request.configs:
                result = self._run_single_benchmark(
                    request.prompt,
                    config,
                    request.parameters
                )
                results.append(result)

        return BenchmarkResponse(
            prompt=request.prompt,
            processing_mode="parallel" if request.parallel_processing else "sequential",
            parameters=request.parameters,
            system_info=system_info,
            results=results,
            timestamp=datetime.now()
        )