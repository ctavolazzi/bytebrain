"""Flask interface for Ollama model benchmarking."""

from flask import Flask, render_template, request, jsonify
from benchmark_ollama import (
    OllamaClient,
    test_model,
    save_benchmark_results,
    get_system_info
)
import os
import json
from datetime import datetime

app = Flask(__name__)

# Initialize the Ollama client
client = OllamaClient()
MODELS = ["wizardlm2", "nemotron-mini", "llama3.2"]

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', models=MODELS)

@app.route('/run_benchmark', methods=['POST'])
def run_benchmark():
    """Run benchmark with the given prompt."""
    prompt = request.json.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt cannot be empty'}), 400

    try:
        # Run benchmarks
        system_info = get_system_info()
        results = []
        for model in MODELS:
            result = test_model(client, model, prompt)
            results.append(result)

        # Save results
        save_benchmark_results(
            results,
            system_info,
            subfolder="user_prompts"
        )

        return jsonify({
            'results': results,
            'system_info': system_info
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get benchmark history."""
    history = []
    history_dir = os.path.join('benchmark_results', 'user_prompts')

    if os.path.exists(history_dir):
        for filename in os.listdir(history_dir):
            if filename.endswith('.json'):
                with open(os.path.join(history_dir, filename), 'r') as f:
                    data = json.load(f)
                    history.append({
                        'timestamp': data['timestamp'],
                        'prompt': data.get('prompt', 'Unknown prompt'),
                        'results': data['results'],
                        'system_info': data['system_info']
                    })

    # Sort by timestamp, newest first
    history.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(history)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('benchmark_results/user_prompts', exist_ok=True)
    app.run(debug=True)