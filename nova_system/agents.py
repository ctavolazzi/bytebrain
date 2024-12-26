from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time

class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        """Initialize the base agent.

        Args:
            name: Name of the agent
        """
        self.name = name
        self.logger = logging.getLogger(f"NovaSystem.{name}")
        self.steps: List[Dict] = []

    def _log_step(self, input_text: str, output_text: str, elapsed_time: float, metadata: Optional[Dict] = None) -> Dict:
        """Log a step in the agent's processing.

        Args:
            input_text: Input text for this step
            output_text: Output text from this step
            elapsed_time: Time taken for this step in seconds
            metadata: Optional additional metadata

        Returns:
            The created step record
        """
        step = {
            "agent_name": self.name,
            "input": input_text,
            "output": output_text,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_ms": elapsed_time * 1000  # Convert to milliseconds
        }
        if metadata:
            step["metadata"] = metadata

        self.steps.append(step)
        self.logger.debug(f"Agent {self.name} completed step: {output_text[:100]}...")
        return step

    @abstractmethod
    async def process(self, context: Dict) -> Dict:
        """Process the given context and return results.

        Args:
            context: The context dictionary containing relevant information

        Returns:
            Dictionary containing processing results
        """
        pass

class PlannerAgent(BaseAgent):
    """Agent responsible for planning response strategies."""

    def __init__(self):
        super().__init__("PlannerAgent")
        self.last_plan = None

    async def process(self, context: Dict) -> Dict:
        """Plan the response strategy based on the input context.

        Args:
            context: Dictionary containing message and conversation history

        Returns:
            Dictionary containing the planned steps
        """
        start_time = time.time()
        self.logger.info("Planning response strategy")

        message = context.get("message", "")
        history = context.get("history", [])

        # TODO: Implement more sophisticated planning logic
        planned_steps = [
            {"type": "analyze", "description": "Analyze user input and context"},
            {"type": "generate", "description": "Generate appropriate response"},
            {"type": "validate", "description": "Validate response against context"}
        ]

        elapsed_time = time.time() - start_time
        step = self._log_step(
            input_text=message,
            output_text=str(planned_steps),
            elapsed_time=elapsed_time,
            metadata={"history_length": len(history)}
        )

        self.last_plan = {
            "planned_steps": planned_steps,
            "step_record": step
        }

        return self.last_plan

class ExecutorAgent(BaseAgent):
    """Agent responsible for executing planned actions."""

    def __init__(self):
        super().__init__("ExecutorAgent")
        self.last_execution = None

    async def process(self, context: Dict) -> Dict:
        """Execute the planned steps.

        Args:
            context: Dictionary containing planned steps and other information

        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        self.logger.info("Executing planned steps")

        steps = context.get("planned_steps", [])
        results = []

        for step in steps:
            # TODO: Implement actual execution logic
            step_start = time.time()
            result = {"step": step, "status": "completed"}
            results.append(result)

            self._log_step(
                input_text=str(step),
                output_text=str(result),
                elapsed_time=time.time() - step_start
            )

        elapsed_time = time.time() - start_time
        final_step = self._log_step(
            input_text=str(steps),
            output_text=str(results),
            elapsed_time=elapsed_time,
            metadata={"total_steps": len(steps)}
        )

        self.last_execution = {
            "execution_results": results,
            "step_record": final_step
        }

        return self.last_execution

class MemoryAgent(BaseAgent):
    """Agent responsible for managing conversation context and memory."""

    def __init__(self):
        super().__init__("MemoryAgent")
        self.conversation_history: List[Dict] = []

    async def process(self, context: Dict) -> Dict:
        """Process and update conversation memory.

        Args:
            context: Dictionary containing new interaction information

        Returns:
            Dictionary containing updated memory state
        """
        start_time = time.time()
        self.logger.info("Updating conversation memory")

        if "message" in context:
            self.conversation_history.append({
                "role": context.get("role", "user"),
                "content": context["message"],
                "metadata": context.get("metadata", {})
            })

        summary = self._generate_context_summary()
        elapsed_time = time.time() - start_time

        step = self._log_step(
            input_text=str(context),
            output_text=str(summary),
            elapsed_time=elapsed_time,
            metadata={"history_length": len(self.conversation_history)}
        )

        return {
            "history": self.conversation_history,
            "context_summary": summary,
            "step_record": step
        }

    def _generate_context_summary(self) -> Dict:
        """Generate a summary of the current conversation context.

        Returns:
            Dictionary containing context summary
        """
        return {
            "total_messages": len(self.conversation_history),
            "last_message": self.conversation_history[-1] if self.conversation_history else None
        }

class AgentOrchestrator:
    """Coordinates between different agents in the system."""

    def __init__(self):
        """Initialize the agent orchestrator."""
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.memory = MemoryAgent()
        self.logger = logging.getLogger("NovaSystem.Orchestrator")

    async def process_turn(self, message: str, metadata: Optional[Dict] = None) -> Dict:
        """Process a complete interaction turn.

        Args:
            message: The input message to process
            metadata: Optional metadata about the interaction

        Returns:
            Dictionary containing the complete turn results
        """
        start_time = time.time()
        context = {
            "message": message,
            "metadata": metadata or {}
        }

        # Update memory first
        memory_result = await self.memory.process(context)
        context.update(memory_result)

        # Plan response strategy
        plan_result = await self.planner.process(context)
        context.update(plan_result)

        # Execute planned steps
        execution_result = await self.executor.process(context)
        context.update(execution_result)

        # Collect all chain steps
        chain_steps = (
            self.memory.steps +
            self.planner.steps +
            self.executor.steps
        )

        context["chain_steps"] = chain_steps
        context["total_time_ms"] = (time.time() - start_time) * 1000

        return context

    def get_agent_states(self) -> Dict:
        """Get the current state of all agents.

        Returns:
            Dictionary containing agent states
        """
        return {
            "memory": self.memory._generate_context_summary(),
            "last_plan": self.planner.last_plan,
            "last_execution": self.executor.last_execution,
            "total_steps": len(self.memory.steps + self.planner.steps + self.executor.steps)
        }