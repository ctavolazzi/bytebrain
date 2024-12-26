import logging
import json
import os
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
import uuid
from ollama_client import OllamaClient

class NovaSystemCore:
    """Core system for managing Nova's operations and interactions."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the Nova System Core.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.conversation_history: List[Dict] = []
        self.session_id = str(uuid.uuid4())
        self.version = "0.1.0"

        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("history", exist_ok=True)

        self.logger.info("NovaSystemCore initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing configuration
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config if it doesn't exist
                default_config = {
                    "model": "wizardlm2",
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "logging": {
                        "level": "INFO",
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    }
                }
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {
                "model": "wizardlm2",
                "temperature": 0.7,
                "max_tokens": 512,
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            }

    def _setup_logging(self):
        """Configure logging for the system."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("NovaSystem")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler
        fh = logging.FileHandler(
            os.path.join(log_dir, f"nova_system_{datetime.now().strftime('%Y%m%d')}.log")
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def _get_system_metrics(self) -> Dict:
        """Get current system performance metrics.

        Returns:
            Dictionary containing system metrics
        """
        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(),
            "mem_usage_mb": process.memory_info().rss / (1024 * 1024),
            "system_cpu_percent": psutil.cpu_percent(),
            "system_memory_percent": psutil.virtual_memory().percent,
            "timestamp": datetime.now().isoformat()
        }

    def _get_git_info(self) -> Dict:
        """Get git repository information.

        Returns:
            Dictionary containing git information
        """
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            return {
                "git_commit": commit,
                "git_branch": branch
            }
        except:
            return {
                "git_commit": "unknown",
                "git_branch": "unknown"
            }

    def _log_interaction(self, role: str, content: str, chain_steps: Optional[List[Dict]] = None, metadata: Optional[Dict] = None) -> Dict:
        """Log an interaction with detailed metadata.

        Args:
            role: Role of the interaction (user/assistant/system)
            content: Content of the interaction
            chain_steps: Optional list of agent chain steps
            metadata: Optional additional metadata

        Returns:
            The created interaction record
        """
        interaction = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": {
                "session_id": self.session_id,
                "resource_usage": self._get_system_metrics(),
                "model_details": {
                    "model_name": self.config.get("model", "llama3.2"),
                    "temperature": self.config.get("temperature", 0.7),
                    "max_tokens": self.config.get("max_tokens", 512)
                },
                "program_info": {
                    "version": self.version,
                    **self._get_git_info()
                },
                "custom_tags": [],
                **(metadata or {})
            }
        }

        if chain_steps:
            interaction["chain_steps"] = chain_steps

        self.conversation_history.append(interaction)
        self.logger.info(f"Interaction logged: {role}")
        return interaction

    async def process_message(self, message: str, chain_steps: Optional[List[Dict]] = None) -> Dict:
        """Process an incoming message and generate a response.

        Args:
            message: The input message to process
            chain_steps: Optional list of agent chain steps

        Returns:
            Dictionary containing the response and metadata
        """
        self.logger.info(f"Processing message: {message}")
        start_time = datetime.now()

        try:
            # Log user message
            user_interaction = self._log_interaction("user", message)

            # Initialize Ollama client
            client = OllamaClient()

            # Prepare model parameters
            model = self.config.get("model", "llama3.2")
            temperature = self.config.get("temperature", 0.7)
            max_tokens = self.config.get("max_tokens", 512)

            # Get response from model
            response = await client.chat(
                model=model,
                messages=[{"role": "user", "content": message}],
                stream=False  # For now, we'll use non-streaming for simplicity
            )

            # Extract response text
            response_text = response["message"]["content"]

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Log assistant response with chain steps and metadata
            assistant_interaction = self._log_interaction(
                "assistant",
                response_text,
                chain_steps=chain_steps,
                metadata={
                    "processing_time_seconds": processing_time,
                    "model_details": {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
            )

            return {
                "response": response_text,
                "user_interaction": user_interaction,
                "assistant_interaction": assistant_interaction,
                "processing_time_seconds": processing_time,
                "success": True
            }

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            self.logger.error(error_msg)

            # Log error as system message
            error_interaction = self._log_interaction(
                "system",
                error_msg,
                metadata={
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )

            return {
                "response": "I apologize, but I encountered an error processing your message.",
                "error": str(e),
                "error_interaction": error_interaction,
                "success": False
            }

    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation history.

        Returns:
            Dictionary containing conversation summary
        """
        return {
            "session_id": self.session_id,
            "total_interactions": len(self.conversation_history),
            "start_time": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "last_update": self.conversation_history[-1]["timestamp"] if self.conversation_history else None,
            "roles_distribution": {
                role: len([x for x in self.conversation_history if x["role"] == role])
                for role in set(x["role"] for x in self.conversation_history)
            }
        }

    def cleanup(self):
        """Clean up resources and close any open files."""
        try:
            # Close logger handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

            # Remove config file if it was created during testing
            if os.path.exists(self.config_path) and self.config_path != "config.json":
                os.remove(self.config_path)

            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise