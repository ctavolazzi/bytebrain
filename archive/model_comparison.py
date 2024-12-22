import json
from ollama_model_benchmarker import OllamaBot

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def compare_responses(prompt, models):
    """Compare responses from different models for the same prompt"""
    bot = OllamaBot()

    print(f"\nPrompt: {prompt}\n")
    for model in models:
        print(f"=== {model} ===")
        response = bot.generate_response(prompt, model=model)
        if response:
            print(response)
            print()
        else:
            print("Error: No response received\n")

if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Use default prompt from config
    prompt = config['default_prompt']
    models = config['models']

    # Run comparison
    compare_responses(prompt, models)
