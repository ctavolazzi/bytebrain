"""Command-line interface for running AutoGen benchmarks."""
import os
import json
import click
import yaml
from typing import Optional
from datetime import datetime

from .models.ag_benchmark import (
    BenchmarkRequest,
    LLMConfig,
    AgentConfig,
    ConversationConfig
)
from .benchmark_runner import BenchmarkRunner

@click.group()
def cli():
    """AutoGen Benchmarking CLI."""
    pass

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('prompt')
@click.option('--parallel', is_flag=True, help='Run benchmarks in parallel')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def run(config_file: str, prompt: str, parallel: bool, output: Optional[str]):
    """Run benchmarks using the specified configuration file and prompt."""
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)

        # Convert configuration to models
        configs = []
        for conv_config in config_data['configurations']:
            agents = []
            for agent_data in conv_config['agents']:
                llm_config = None
                if 'llm_config' in agent_data:
                    llm_config = LLMConfig(**agent_data['llm_config'])

                agent = AgentConfig(
                    name=agent_data['name'],
                    type=agent_data['type'],
                    llm_config=llm_config,
                    system_message=agent_data.get('system_message', '')
                )
                agents.append(agent)

            config = ConversationConfig(
                name=conv_config['name'],
                agents=agents,
                initiator=conv_config['initiator'],
                max_rounds=conv_config.get('max_rounds', 10),
                description=conv_config.get('description', '')
            )
            configs.append(config)

        # Create benchmark request
        request = BenchmarkRequest(
            prompt=prompt,
            configs=configs,
            parallel_processing=parallel
        )

        # Run benchmark
        runner = BenchmarkRunner()
        response = runner.run_benchmark(request)

        # Save results
        if output:
            with open(output, 'w') as f:
                json.dump(response.dict(), f, indent=2, default=str)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(response.dict(), indent=2, default=str))

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def history(directory: str):
    """List benchmark history from the specified directory."""
    try:
        results = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                with open(os.path.join(directory, filename), 'r') as f:
                    data = json.load(f)
                    results.append({
                        'id': filename.replace('benchmark_', '').replace('.json', ''),
                        'timestamp': data['timestamp'],
                        'prompt': data['prompt'],
                        'success_rate': sum(1 for r in data['results'] if r['success']) / len(data['results'])
                    })

        # Sort by timestamp
        results.sort(key=lambda x: x['timestamp'], reverse=True)

        # Print results
        for result in results:
            click.echo(
                f"ID: {result['id']}\n"
                f"Timestamp: {result['timestamp']}\n"
                f"Prompt: {result['prompt']}\n"
                f"Success Rate: {result['success_rate']:.2%}\n"
            )

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
def create_config():
    """Create a sample configuration file."""
    sample_config = {
        "configurations": [
            {
                "name": "basic_conversation",
                "description": "Basic conversation between assistant and user",
                "agents": [
                    {
                        "name": "assistant",
                        "type": "assistant",
                        "llm_config": {
                            "provider": "openai",
                            "model": "gpt-3.5-turbo",
                            "api_key": "${OPENAI_API_KEY}"
                        },
                        "system_message": "You are a helpful AI assistant."
                    },
                    {
                        "name": "user",
                        "type": "user_proxy",
                        "system_message": "You are a user seeking assistance."
                    }
                ],
                "initiator": "user",
                "max_rounds": 5
            }
        ]
    }

    with open('benchmark_config_sample.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)

    click.echo("Sample configuration saved to benchmark_config_sample.yaml")

if __name__ == '__main__':
    cli()