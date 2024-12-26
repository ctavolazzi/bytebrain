from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
import logging
import json
import asyncio
from sse_starlette.sse import EventSourceResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.models.benchmark import BenchmarkRequest, BenchmarkResponse
from app.services.benchmark import BenchmarkService
from app.services.storage import StorageService

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))

# Initialize services
benchmark_service = BenchmarkService()
storage_service = StorageService()

# Create a shared queue for streaming updates
benchmark_updates = asyncio.Queue()

@router.get("/stream")
async def stream_benchmark_updates():
    """Stream real-time benchmark updates using Server-Sent Events."""
    async def event_generator():
        try:
            while True:
                update = await benchmark_updates.get()
                if update is None:  # None is our signal to stop
                    break
                yield {
                    "data": json.dumps(update)
                }
        except asyncio.CancelledError:
            logger.info("Client disconnected from event stream")
        except Exception as e:
            logger.error(f"Error in event stream: {e}")
            raise

    return EventSourceResponse(event_generator())

@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """
    Run benchmarks on selected models with the given prompt.

    Args:
        request: BenchmarkRequest containing prompt, models, and parameters

    Returns:
        BenchmarkResponse with results and timing information

    Raises:
        HTTPException: If benchmark execution fails
    """
    try:
        result = await benchmark_service.run_benchmark(
            prompt=request.prompt,
            models=request.models,
            parameters=request.parameters,
            update_queue=benchmark_updates  # Pass the queue for streaming updates
        )
        await storage_service.save_benchmark(result)
        # Signal end of streaming
        await benchmark_updates.put(None)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )

@router.get("/history")
async def get_benchmark_history(limit: int = 50):
    """Get the history of benchmark runs with validated limits."""
    try:
        # Get the benchmarks directory path
        benchmarks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "benchmarks")

        # List all benchmark files
        benchmark_files = []
        if os.path.exists(benchmarks_dir):
            for filename in sorted(os.listdir(benchmarks_dir), reverse=True):
                if filename.startswith('benchmark_') and filename.endswith('.json'):
                    file_path = os.path.join(benchmarks_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            # Extract benchmark ID from filename
                            benchmark_id = filename.replace('benchmark_', '').replace('.json', '')
                            benchmark_files.append({
                                "id": benchmark_id,
                                "timestamp": data.get("timestamp"),
                                "prompt": data.get("prompt")
                            })
                            if len(benchmark_files) >= limit:
                                break
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing {filename}, skipping...")
                        continue

        return benchmark_files
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )

@router.get("/history/{benchmark_id}")
async def get_benchmark_by_id(benchmark_id: str):
    """Get a specific benchmark result by ID."""
    logger.info(f"Fetching benchmark with ID: {benchmark_id}")
    try:
        # Construct the file path using the benchmark ID
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "benchmarks", f"benchmark_{benchmark_id}.json")

        if not os.path.exists(file_path):
            logger.warning(f"Benchmark file not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Benchmark {benchmark_id} not found"
            )

        # Read the benchmark file directly
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
        # Get list of recent benchmarks for display
        logger.info("Fetching recent benchmarks for UI")
        recent_benchmarks = await get_benchmark_history(limit=5)
        logger.info(f"Found {len(recent_benchmarks)} recent benchmarks")
        return templates.TemplateResponse(
            "benchmark.html",
            {
                "request": request,
                "recent_benchmarks": recent_benchmarks
            }
        )
    except Exception as e:
        logger.error(f"Error rendering UI: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render UI: {str(e)}"
        )
