from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Create FastAPI app
app = FastAPI(
    title="Ollama Benchmark",
    description="A modern benchmarking tool for Ollama language models",
    version="0.1.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Import and include routers
from app.api.benchmark import router as benchmark_router
from app.api.models import router as models_router

app.include_router(benchmark_router, prefix="/api/benchmark", tags=["benchmark"])
app.include_router(models_router, prefix="/api/models", tags=["models"])

# Root endpoint to serve the frontend
@app.get("/", response_class=HTMLResponse)
async def get_home():
    # Create necessary directories
    os.makedirs("benchmark_results/user_prompts", exist_ok=True)

    # Return index.html
    return templates.TemplateResponse("index.html", {"request": {}})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)