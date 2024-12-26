"""Autogen client service for managing agent interactions."""
import asyncio
import time
from typing import Dict, List, Optional, Any
import autogen

from ..models.agent_config import AgentConfig, ConversationConfig, LLMConfig

class AutogenError(Exception):
    """Custom exception for Autogen-related errors."""
    pass

class AutogenClient:
    """Client for managing Autogen agents and conversations."""

    def __init__(self):
        """Initialize the Autogen client."""
        self.agents = {}
        self.conversations = {}

    def _create_llm_config(self, llm_config: LLMConfig) -> Dict[str, Any]:
        """Create LLM configuration for Autogen."""
        if llm_config.provider == "ollama":
            return {
                "model": llm_config.model,
                "base_url": llm_config.base_url or "http://localhost:11434",
                "api_type": "ollama"
            }

        # Default OpenAI-style config for other providers
        config = {
            "config_list": [{
                "model": llm_config.model,
            }]
        }

        if llm_config.provider == "openai":
            if not llm_config.api_key:
                raise AutogenError("OpenAI API key is required")
            config["config_list"][0].update({
                "api_key": llm_config.api_key,
            })
        elif llm_config.provider == "anthropic":
            if not llm_config.api_key:
                raise AutogenError("Anthropic API key is required")
            config["config_list"][0].update({
                "api_key": llm_config.api_key,
                "base_url": llm_config.base_url or "https://api.anthropic.com/v1",
            })
        elif llm_config.provider == "azure":
            if not (llm_config.api_key and llm_config.base_url):
                raise AutogenError("Azure API key and base URL are required")
            config["config_list"][0].update({
                "api_key": llm_config.api_key,
                "base_url": llm_config.base_url,
                "api_type": "azure",
            })
        elif llm_config.provider == "local":
            config["config_list"][0].update({
                "base_url": llm_config.base_url or "http://localhost:11434",
            })

        # Add any additional configuration
        if llm_config.additional_config:
            config["config_list"][0].update(llm_config.additional_config)

        return config

    def create_agent(self, config: AgentConfig) -> Any:
        """Create an Autogen agent based on configuration."""
        try:
            # Create the appropriate agent type
            if config.type == "assistant":
                agent = autogen.AssistantAgent(
                    name=config.name,
                    system_message=config.system_message,
                    llm_config=self._create_llm_config(config.llm_config) if config.llm_config else None
                )
            elif config.type == "user_proxy":
                agent = autogen.UserProxyAgent(
                    name=config.name,
                    system_message=config.system_message,
                    human_input_mode="NEVER",
                    code_execution_config={"use_docker": False}
                )
            else:
                raise AutogenError(f"Unsupported agent type: {config.type}")

            self.agents[config.name] = agent
            return agent

        except Exception as e:
            raise AutogenError(f"Failed to create agent: {str(e)}")

    async def run_conversation(
        self,
        config: ConversationConfig,
        prompt: str,
        update_queue: Optional[asyncio.Queue] = None
    ) -> Dict:
        """Run a conversation between Autogen agents."""
        try:
            # Create agents if they don't exist
            for agent_config in config.agents:
                if agent_config.name not in self.agents:
                    self.create_agent(agent_config)

            # Get initiator and recipient
            initiator = self.agents.get(config.initiator)
            if not initiator:
                raise AutogenError(f"Initiator agent '{config.initiator}' not found")

            # Get the first non-initiator agent as recipient
            recipient = next((agent for name, agent in self.agents.items() if name != config.initiator), None)
            if not recipient:
                raise AutogenError("No recipient agent found")

            # Track metrics
            start_time = time.time()
            start_monotonic = time.monotonic()
            messages_sent = 0
            total_tokens = 0

            # Start conversation
            if update_queue:
                await update_queue.put({
                    "status": "starting",
                    "message": f"Starting conversation with initiator: {config.initiator}"
                })

            # Initiate the conversation
            initiator.initiate_chat(
                recipient,
                message=prompt
            )

            # Calculate final metrics
            end_time = time.monotonic()
            total_time = end_time - start_monotonic

            # Get conversation results
            result = {
                "messages": initiator.chat_messages[recipient],
                "timing": {
                    "total_time": total_time,
                },
                "metrics": {
                    "messages_sent": len(initiator.chat_messages[recipient]),
                    "total_tokens": total_tokens,  # This would need to be calculated based on the tokenizer
                },
                "success": True
            }

            if update_queue:
                await update_queue.put({
                    "status": "completed",
                    "total_time": total_time,
                    "total_messages": len(initiator.chat_messages[recipient])
                })

            return result

        except Exception as e:
            if update_queue:
                await update_queue.put({
                    "status": "error",
                    "error": str(e)
                })
            raise AutogenError(f"Conversation failed: {str(e)}")

    def reset(self):
        """Reset all agents and conversations."""
        self.agents = {}
        self.conversations = {}