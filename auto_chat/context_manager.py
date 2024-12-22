import json
import os
from datetime import datetime
from typing import Dict, Optional
import re
import logging

class ContextManager:
    """Manages the context for auto chat conversations."""

    def __init__(self, session_manager, bot, config: Dict, debug_session_name: Optional[str] = None):
        """Initialize the context manager with a session manager and configuration."""
        self.session_id = debug_session_name or session_manager.session_id
        self.session_dir = os.path.join("history", "sessions", self.session_id)
        self.context_dir = os.path.join("history", "contexts", self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.context_dir, exist_ok=True)
        self.bot = bot
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.initial_user_prompt = None
        self.is_first_context = True
        self.last_two_responses = []  # Track last two responses

    def _save_context(self, context: Dict, timestamp: str):
        """Save context to the session folder."""
        context_file = os.path.join(self.context_dir, f"context_{timestamp}.json")
        with open(context_file, 'w') as f:
            json.dump(context, f, indent=2)
        self.logger.info(f"Saved context to {context_file}")

    def generate_context(self, initial_prompt: Optional[str] = None, last_response: Optional[str] = None) -> Dict:
        """Generate context for the next model using llama3.2."""
        try:
            context_model = self.config["auto_chat"]["context_model"]
            self.logger.info(f"Using {context_model} to generate context")

            if self.is_first_context:
                # First context from user prompt
                self.initial_user_prompt = initial_prompt
                context_prompt = f"You are generating the first context for a chat between AI models. The user inputted: {initial_prompt}\nGenerate context to start this conversation."
                self.is_first_context = False
            else:
                # Update last two responses
                if last_response:
                    self.last_two_responses.append(last_response)
                    if len(self.last_two_responses) > 2:
                        self.last_two_responses.pop(0)

                # Generate context from last two responses
                if len(self.last_two_responses) == 2:
                    context_prompt = f"Original user input: {self.initial_user_prompt}\nPrevious two responses:\n1. {self.last_two_responses[0]}\n2. {self.last_two_responses[1]}"
                else:
                    context_prompt = f"Original user input: {self.initial_user_prompt}\nPrevious response: {last_response}"

            # Get context from llama3.2
            print(f"\n[CONTEXT BOT - {context_model}]:")
            self.logger.info(f"CONTEXT BOT - {context_model} starting context generation")
            generated_context = self.bot.stream_response(
                prompt=context_prompt,
                model=context_model
            )
            context_text = ""
            for chunk in generated_context:
                print(chunk, end='', flush=True)
                context_text += chunk
            print()  # New line after context generation
            self.logger.info(f"CONTEXT BOT - {context_model} finished context generation")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            context = {
                "initial_prompt": context_text,
                "original_user_prompt": self.initial_user_prompt,
                "timestamp": timestamp,
                "context_prompt": context_prompt
            }

            self._save_context(context, timestamp)
            self.logger.info(f"Generated context: {json.dumps(context, indent=2)}")
            return context

        except Exception as e:
            self.logger.error(f"Error generating context: {str(e)}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = {
                "initial_prompt": last_response or self.initial_user_prompt,
                "original_user_prompt": self.initial_user_prompt,
                "timestamp": timestamp,
                "error": str(e)
            }
            self._save_context(fallback, timestamp)
            self.logger.info(f"Generated fallback context: {json.dumps(fallback, indent=2)}")
            return fallback

    def update_context(self, last_response: str) -> Dict:
        """Update context based on the last response."""
        self.logger.info(f"Updating context with response: {last_response}")
        context = self.generate_context(last_response=last_response)
        self.logger.info(f"Updated context: {json.dumps(context, indent=2)}")
        return context
