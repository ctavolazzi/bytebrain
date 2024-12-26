from fastapi import APIRouter, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
import logging
import json
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
import asyncio

from ..models.benchmark import BenchmarkRequest, BenchmarkResponse
from ..services.benchmark import BenchmarkService
from ..services.stream import stream_queue

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up directories
PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
BENCHMARKS_DIR = os.path.join(PACKAGE_DIR, "data", "benchmarks")
PROMPT_LIBRARY_PATH = os.path.join(PACKAGE_DIR, "data", "prompt_library.json")
os.makedirs(BENCHMARKS_DIR, exist_ok=True)
logger.info(f"Using benchmarks directory: {BENCHMARKS_DIR}")

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(PACKAGE_DIR, "templates"))

# Initialize services
benchmark_service = BenchmarkService()

class SavePromptRequest(BaseModel):
    prompt: str
    name: Optional[str] = None
    category: str = "Custom"

@router.get("/prompts")
async def get_prompt_library():
    """Get the list of predefined prompts from the prompt library."""
    try:
        if os.path.exists(PROMPT_LIBRARY_PATH):
            with open(PROMPT_LIBRARY_PATH, 'r') as f:
                prompt_library = json.load(f)
                return prompt_library
        else:
            logger.warning("Prompt library file not found")
            return {"default_prompts": [], "user_prompts": []}
    except Exception as e:
        logger.error(f"Failed to load prompt library: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load prompt library: {str(e)}"
        )

@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """Run benchmarks on selected models with the given prompt."""
    logger.info("\n" + "="*80)
    logger.info("🚀 BENCHMARK RUN INITIATED")
    logger.info(f"📝 Prompt: {request.prompt}")
    logger.info(f"🤖 Models: {request.models}")
    logger.info(f"🔄 Processing Mode: {'⚡️ Parallel' if request.parallel_processing else '📝 Sequential'}")
    if request.parameters:
        logger.info(f"⚙️  Parameters: {request.parameters}")
    logger.info("="*80)

    try:
        result = await benchmark_service.run_benchmark(
            prompt=request.prompt,
            models=request.models,
            parameters=request.parameters,
            parallel_processing=request.parallel_processing
        )

        try:
            # Save benchmark to file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(BENCHMARKS_DIR, f"benchmark_{timestamp}.json")
            with open(file_path, 'w') as f:
                json.dump(result.dict(), f, default=str, indent=2)
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
        logger.error(f"Stack trace: ", exc_info=True)  # This will log the full stack trace
        logger.info("="*80 + "\n")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}"
        )

@router.get("/history")
async def get_benchmark_history(limit: int = 50):
    """Get the history of benchmark runs with validated limits."""
    try:
        # List all benchmark files
        benchmark_files = []
        if os.path.exists(BENCHMARKS_DIR):
            for filename in sorted(os.listdir(BENCHMARKS_DIR), reverse=True):
                if filename.startswith('benchmark_') and filename.endswith('.json'):
                    file_path = os.path.join(BENCHMARKS_DIR, filename)
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
        logger.error(f"Failed to retrieve history: {e}")
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
        file_path = os.path.join(BENCHMARKS_DIR, f"benchmark_{benchmark_id}.json")

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

@router.post("/prompts/save")
async def save_prompt(prompt_data: SavePromptRequest):
    """Save a new prompt to the prompt library."""
    try:
        # Load existing prompts
        with open(PROMPT_LIBRARY_PATH, 'r') as f:
            library = json.load(f)

        # Check if prompt already exists in user_prompts
        existing_prompts = [p["prompt"] for p in library.get("user_prompts", [])]
        if prompt_data.prompt in existing_prompts:
            logger.info(f"Prompt already exists, skipping save: {prompt_data.prompt[:50]}...")
            return {"status": "skipped", "message": "Prompt already exists"}

        # Add new prompt
        new_prompt = {
            "name": prompt_data.name or f"Custom Prompt {len(library.get('user_prompts', [])) + 1}",
            "prompt": prompt_data.prompt,
            "category": prompt_data.category or "Custom"
        }

        if "user_prompts" not in library:
            library["user_prompts"] = []

        library["user_prompts"].append(new_prompt)

        # Save updated library
        with open(PROMPT_LIBRARY_PATH, 'w') as f:
            json.dump(library, f, indent=2)

        logger.info(f"Saving new prompt: prompt='{prompt_data.prompt[:50]}...' name={prompt_data.name} category='{prompt_data.category}'")
        return {"status": "success", "message": "Prompt saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/prompts/user/{prompt_name}")
async def delete_user_prompt(prompt_name: str):
    """Delete a specific user prompt from the library."""
    try:
        with open(PROMPT_LIBRARY_PATH, 'r') as f:
            library = json.load(f)

        library["user_prompts"] = [p for p in library["user_prompts"] if p["name"] != prompt_name]

        with open(PROMPT_LIBRARY_PATH, 'w') as f:
            json.dump(library, f, indent=2)

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to delete prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/prompts/user")
async def clear_user_prompts():
    """Clear all user prompts from the library."""
    try:
        with open(PROMPT_LIBRARY_PATH, 'r') as f:
            library = json.load(f)

        library["user_prompts"] = []

        with open(PROMPT_LIBRARY_PATH, 'w') as f:
            json.dump(library, f, indent=2)

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to clear prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream")
async def stream_benchmark(request: Request):
    """Stream benchmark updates to the client."""
    async def event_generator():
        try:
            while True:
                try:
                    # Get update from queue with timeout
                    update = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                    logger.info(f"Sending update: {update}")  # Add logging
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