"""FastAPI endpoints for the AutoGen benchmarking system."""
import os
import json
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .models.ag_benchmark import BenchmarkRequest, BenchmarkResponse
from .benchmark_runner import BenchmarkRunner

app = FastAPI(title="AutoGen Benchmark API")
benchmark_runner = BenchmarkRunner()

# Store for active WebSocket connections
active_connections: List[WebSocket] = []

@app.post("/api/benchmarks/run")
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """Run a benchmark with the given configuration."""
    try:
        response = benchmark_runner.run_benchmark(request)

        # Save benchmark results
        save_benchmark_results(response)

        # Notify WebSocket clients
        await notify_clients({
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "results": response.dict()
        })

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmarks/history")
async def get_benchmark_history() -> List[dict]:
    """Get history of all benchmark runs."""
    try:
        history = []
        history_dir = os.path.join(os.path.dirname(__file__), "data", "benchmarks")
        os.makedirs(history_dir, exist_ok=True)

        for filename in os.listdir(history_dir):
            if filename.endswith(".json"):
                with open(os.path.join(history_dir, filename), "r") as f:
                    history.append(json.load(f))

        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmarks/history/{benchmark_id}")
async def get_benchmark_details(benchmark_id: str) -> dict:
    """Get details of a specific benchmark run."""
    try:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "benchmarks",
            f"benchmark_{benchmark_id}.json"
        )

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Benchmark not found")

        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/benchmarks/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time benchmark updates."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)

async def notify_clients(message: dict):
    """Send update to all connected WebSocket clients."""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            active_connections.remove(connection)

def save_benchmark_results(response: BenchmarkResponse):
    """Save benchmark results to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "benchmarks",
        f"benchmark_{timestamp}.json"
    )

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(response.dict(), f, indent=2, default=str)