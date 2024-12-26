"""NovaSystem Benchmarking - A modular benchmarking system for LLM models using Ollama."""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import os

# Create FastAPI app
app = FastAPI(
    title="Ollama Benchmarker",
    description="A FastAPI-based benchmarking tool for local LLMs via Ollama",
    version="0.1.0"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Import routers
from .routers import benchmarks

# Include routers
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])

@app.get("/")
async def root():
    """Redirect root to benchmarking UI"""
    return RedirectResponse(url="/api/benchmarks/")

# Export key components
from .services.ollama_client import OllamaClient, OllamaError

__all__ = ["app", "OllamaClient", "OllamaError"]
