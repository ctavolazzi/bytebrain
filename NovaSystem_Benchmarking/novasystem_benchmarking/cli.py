"""Command-line interface for NovaSystem Benchmarking."""
import click
import uvicorn
from . import app

@click.group()
def main():
    """NovaSystem Benchmarking CLI."""
    pass

@main.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the benchmarking server."""
    click.echo(f"Starting server on {host}:{port}")
    uvicorn.run(
        "novasystem_benchmarking:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == '__main__':
    main()