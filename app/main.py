from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
import uvicorn
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
from app.routers import benchmarks

# Include routers
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["benchmarks"])

@app.on_event("startup")
async def startup():
    """Initialize FastAPI cache on startup"""
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")

@app.get("/")
async def root():
    """Redirect root to benchmarking UI"""
    return RedirectResponse(url="/api/benchmarks/")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)