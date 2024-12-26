#!/usr/bin/env python3
import asyncio
import argparse
import sys
from typing import Optional

from auto_chat.auto_command import AutoCommandHandler

class NovaChat:
    """Interactive chat interface for NovaSystem."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the chat interface.

        Args:
            config_path: Optional path to configuration file
        """
        self.handler = AutoCommandHandler(config_path=config_path or "config.json")
        self.running = False

    async def process_input(self, user_input: str, stream: bool = True):
        """Process user input and display response.

        Args:
            user_input: The user's input message
            stream: Whether to stream the response
        """
        if user_input.lower() in ['/quit', '/exit']:
            self.running = False
            return

        if user_input.lower() == '/info':
            info = self.handler.get_session_info()
            print("\nSession Information:")
            print(f"Model Version: {info['model_version']}")
            print(f"Total Interactions: {info['session']['interaction_count']}")
            print(f"Session ID: {info['session']['session_id']}")
            return

        try:
            response = await self.handler.process_message(user_input, stream=stream)

            if stream:
                print("\nBot:", end=" ", flush=True)
                async for chunk in response:
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                print("\nBot:", response, "\n")

        except Exception as e:
            print(f"\nError: {str(e)}\n")

    async def run(self, stream: bool = True):
        """Run the interactive chat loop.

        Args:
            stream: Whether to stream responses
        """
        self.running = True
        print("\nWelcome to NovaChat!")
        print("Commands:")
        print("  /info - Show session information")
        print("  /quit or /exit - Exit the chat")
        print("\nEnter your message (press Ctrl+C to exit):")

        while self.running:
            try:
                user_input = input("\nYou: ")
                if user_input.strip():
                    await self.process_input(user_input, stream=stream)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                break

        self.handler.close()

def main():
    """Main entry point for the chat application."""
    parser = argparse.ArgumentParser(description="NovaSystem Chat Interface")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--no-stream", action="store_true", help="Disable response streaming")
    args = parser.parse_args()

    chat = NovaChat(config_path=args.config)

    try:
        asyncio.run(chat.run(stream=not args.no_stream))
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()