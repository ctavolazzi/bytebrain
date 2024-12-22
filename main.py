import json
import logging
import time
import sys
import os
from datetime import datetime
from auto_chat import AutoCommandHandler
from auto_chat.session_manager import SessionManager

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_logging():
    """Setup basic logging to console and file"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Setup logging with timestamp in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/ollama_bot_{timestamp}.log'

    # Configure logging to both file and console with immediate flush
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Log initialization
    logging.info("="*50)
    logging.info("INITIALIZING OLLAMA BOT")
    logging.info("="*50)

    return log_file

def load_config(config_path='config.json'):
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return None

def save_prompt_to_history(prompt):
    """Save user prompt to history file"""
    # Create history directory if it doesn't exist
    if not os.path.exists('history'):
        os.makedirs('history')

    # Save to single history file
    with open('history/prompts.txt', 'a') as f:
        f.write(prompt + '\n')

def get_or_create_session() -> SessionManager:
    """Get current session or create a new one."""
    current_session_id = SessionManager.get_current_session()
    if current_session_id:
        return SessionManager(current_session_id)
    return SessionManager.create_new_session()

def handle_command(command: str) -> None:
    """Handle a command from the user."""
    if command.lower() in ['exit', 'quit', 'q']:
        print("\nExiting...")
        sys.exit(0)

    # Get or create session
    session = get_or_create_session()

    # Create command handler
    handler = AutoCommandHandler(session_manager=session)

    # Start the command handler with the user's input
    handler.start(command)

def main():
    """Main function to run the bot"""
    # Setup logging
    setup_logging()

    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Exiting.")
        return

    print("\nOllama Bot initialized. Enter a command or prompt (type 'exit' to quit):")

    while True:
        try:
            # Get user input
            command = input("\n> ")

            # Handle the command
            handle_command(command)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logging.error(f"Error in main loop: {str(e)}")
            continue

if __name__ == "__main__":
    main()
