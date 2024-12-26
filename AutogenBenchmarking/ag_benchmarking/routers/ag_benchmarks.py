"""FastAPI router for benchmark endpoints."""
from fastapi import APIRouter, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
import logging
import json
from datetime import datetime
from typing import Optional, List
import asyncio

from ..models.ag_benchmark import BenchmarkRequest, BenchmarkResponse
from ..services.ag_benchmark import BenchmarkService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up directories
PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
BENCHMARKS_DIR = os.path.join(PACKAGE_DIR, "data", "benchmarks")
PROMPT_LIBRARY_PATH = os.path.join(PACKAGE_DIR, "data", "prompt_library", "prompts.json")
os.makedirs(BENCHMARKS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROMPT_LIBRARY_PATH), exist_ok=True)

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(PACKAGE_DIR, "templates"))

# Initialize services
benchmark_service = BenchmarkService()

# Create a shared queue for streaming updates
stream_queue = asyncio.Queue()

@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """Run benchmarks on selected conversation configurations with the given prompt."""
    logger.info("\n" + "="*80)
    logger.info("🚀 BENCHMARK RUN INITIATED")
    logger.info(f"📝 Prompt: {request.prompt}")
    logger.info(f"🤖 Configurations: {[c.name for c in request.configs]}")
    logger.info(f"🔄 Processing Mode: {'⚡️ Parallel' if request.parallel_processing else '📝 Sequential'}")
    if request.parameters:
        logger.info(f"⚙️  Parameters: {request.parameters}")
    logger.info("="*80)

    try:
        logger.debug("Creating benchmark service...")
        result = await benchmark_service.run_benchmark(
            prompt=request.prompt,
            configs=request.configs,
            parallel_processing=request.parallel_processing,
            parameters=request.parameters
        )
        logger.debug("Benchmark service completed successfully")

        try:
            # Save benchmark to file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(BENCHMARKS_DIR, f"benchmark_{timestamp}.json")
            with open(file_path, 'w') as f:
                json.dump(result, f, default=str, indent=2)
            logger.info(f"💾 Saved benchmark results to {file_path}")
        except Exception as save_error:
            logger.error(f"❌ Failed to save benchmark results: {save_error}")
            # Continue even if save fails - we still want to return the results
            pass

        logger.info("="*80)
        logger.info("✅ BENCHMARK RUN COMPLETED")
        logger.info("="*80 + "\n")
        return result
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        logger.error(f"Request data: {request.dict()}")
        logger.info("="*80 + "\n")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )

@router.get("/history")
async def get_benchmark_history(limit: int = 50):
    """Get history of benchmark runs."""
    try:
        benchmarks = []
        for filename in sorted(os.listdir(BENCHMARKS_DIR), reverse=True)[:limit]:
            if filename.endswith('.json'):
                file_path = os.path.join(BENCHMARKS_DIR, filename)
                with open(file_path, 'r') as f:
                    benchmark_data = json.load(f)
                    benchmark_data['id'] = filename.replace('benchmark_', '').replace('.json', '')
                    benchmarks.append(benchmark_data)
        return benchmarks
    except Exception as e:
        logger.error(f"Failed to get benchmark history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get benchmark history: {str(e)}"
        )

@router.get("/history/{benchmark_id}")
async def get_benchmark_by_id(benchmark_id: str):
    """Get a specific benchmark result by ID."""
    logger.info(f"Fetching benchmark with ID: {benchmark_id}")
    try:
        file_path = os.path.join(BENCHMARKS_DIR, f"benchmark_{benchmark_id}.json")

        if not os.path.exists(file_path):
            logger.warning(f"Benchmark file not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Benchmark {benchmark_id} not found"
            )

        with open(file_path, 'r') as f:
            benchmark_data = json.load(f)
            logger.info(f"Successfully loaded benchmark {benchmark_id}")
            return benchmark_data

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing benchmark file {benchmark_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing benchmark file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error retrieving benchmark {benchmark_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve benchmark: {str(e)}"
        )

@router.get("/", response_class=HTMLResponse)
async def get_benchmark_ui(request: Request):
    """Render the benchmarking UI."""
    try:
        logger.info("Fetching recent benchmarks for UI")
        recent_benchmarks = await get_benchmark_history(limit=50)
        logger.info(f"Found {len(recent_benchmarks)} recent benchmarks")
        return templates.TemplateResponse(
            "ag_benchmark.html",
            {
                "request": request,
                "benchmarks": recent_benchmarks
            }
        )
    except Exception as e:
        logger.error(f"Error rendering UI: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render UI: {str(e)}"
        )

@router.get("/stream")
async def stream_benchmark(request: Request):
    """Stream benchmark updates to the client."""
    async def event_generator():
        try:
            while True:
                try:
                    # Get update from queue with timeout
                    update = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                    logger.info(f"Sending update: {update}")
                    yield f"data: {json.dumps(update)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield "data: {}\n\n"
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    break
        except Exception as e:
            logger.error(f"Event generator error: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )