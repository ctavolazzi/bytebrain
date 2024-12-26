import os
import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import ollama

def create_ollama_config():
    """Create a configuration for Ollama"""
    return {
        "config_list": [{
            "model": "llama3.2",
            "api_base": "http://localhost:11434/api",
            "api_type": "ollama",
            "api_key": "ollama"  # Ollama doesn't need an API key, but config requires this field
        }],
        "temperature": 0.7,
        "timeout": 60
    }

def setup_agents():
    """Set up the Autogen agents with Ollama configuration"""
    # Configure the code execution
    code_execution_config = {
        "work_dir": "workspace",  # Local directory for code execution
        "use_docker": False,  # We'll run code locally instead of in Docker
    }

    # Create the assistant and user proxy agents
    assistant = AssistantAgent(
        name="assistant",
        llm_config=create_ollama_config(),
        system_message="""You are a helpful AI assistant that can write and execute Python code.
        When writing code, make sure it's well-documented and follows best practices.
        You specialize in creating clean, simple, and effective user interfaces."""
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",  # Set to "ALWAYS" if you want to verify code before execution
        code_execution_config=code_execution_config
    )

    return assistant, user_proxy

def test_ollama_connection():
    """Test basic Ollama functionality"""
    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{
                'role': 'user',
                'content': 'Hi, this is a test message. Please respond briefly.'
            }]
        )
        print("Test Response:", response['message']['content'])
        return True
    except Exception as e:
        print(f"Error testing Ollama connection: {e}")
        return False

def run_autogen_chat(message: str):
    """Run an Autogen chat session with the given message"""
    # Create the workspace directory if it doesn't exist
    os.makedirs("workspace", exist_ok=True)

    # Set up the agents
    assistant, user_proxy = setup_agents()

    try:
        # Initiate the chat
        user_proxy.initiate_chat(
            assistant,
            message=message
        )
    except Exception as e:
        print(f"Error during chat: {e}")
        print("Full error details:", str(e))  # Added more detailed error logging

if __name__ == "__main__":
    # First test the Ollama connection
    print("Testing Ollama connection...")
    if test_ollama_connection():
        print("Ollama test succeeded, starting Autogen chat...")

        # Request to create a Streamlit interface
        test_message = """Create a Streamlit interface for our chat system. Here's the exact code structure needed:

        1. First, create a file called 'app.py' with these imports:
           - streamlit
           - ollama
           - any other necessary imports

        2. Create a simple layout with:
           - A title at the top
           - A text input box for the user's message
           - A submit button
           - A area to display the chat history

        3. Implement the basic chat functionality:
           - Use ollama.chat() to send messages to the model
           - Store chat history in session state
           - Display messages in a clean format

        Make the code simple, clean, and well-commented. Save it as 'app.py' in the workspace directory.
        The app should be ready to run with 'streamlit run app.py'."""

        run_autogen_chat(test_message)
    else:
        print("Ollama test failed. Please make sure Ollama is running and a model is pulled.")