"""AutogenBenchmarking - A modular benchmarking system for Autogen agents and conversations."""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import os

# Create FastAPI app
app = FastAPI(
    title="Autogen Benchmarker",
    description="A FastAPI-based benchmarking tool for Autogen agents and conversations",
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
from .routers import ag_benchmarks

# Include routers
app.include_router(ag_benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])
app.include_router(ag_benchmarks.router, tags=["benchmarks"])

@app.get("/")
async def root():
    """Redirect to the benchmarking UI."""
    return RedirectResponse(url="/api/benchmarks/")

# Export key components
from .services.ag_client import AutogenClient, AutogenError

__all__ = ["app", "AutogenClient", "AutogenError"]
