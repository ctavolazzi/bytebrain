from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional, Dict
import os
import json
import logging
from datetime import datetime

from app.models.benchmark import BenchmarkResponse

# Set up logging
logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        # Default to JSON storage
        self.use_mongo = False

        # Use the app/data/benchmarks directory
        self.json_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app", "data", "benchmarks")
        self.json_file = os.path.join(self.json_dir, "benchmark_results.json")

        logger.info(f"Initializing StorageService with JSON directory: {self.json_dir}")

        # Create benchmarks directory if it doesn't exist
        os.makedirs(self.json_dir, exist_ok=True)

        # Try MongoDB connection only if explicitly configured
        mongo_url = os.getenv("MONGODB_URL")
        if mongo_url:
            try:
                self.client = AsyncIOMotorClient(mongo_url)
                self.db = self.client.ollama_benchmarks
                self.collection = self.db.benchmark_results
                self.use_mongo = True
                logger.info("Successfully connected to MongoDB")
            except Exception as e:
                logger.error(f"MongoDB connection failed: {e}. Using JSON storage.")
                self.use_mongo = False

    def _convert_timestamps(self, result: Dict) -> Dict:
        """Convert timestamp strings to datetime objects."""
        if isinstance(result.get('timestamp'), str):
            result['timestamp'] = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
        return result

    def _load_existing_results(self) -> List[dict]:
        """Load and merge results from all JSON files in the directory."""
        results = []
        if os.path.exists(self.json_dir):
            logger.info(f"Loading benchmark files from {self.json_dir}")
            for filename in os.listdir(self.json_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.json_dir, filename)
                    logger.debug(f"Reading benchmark file: {filename}")
                    try:
                        with open(file_path, 'r') as f:
                            file_results = json.load(f)
                            # Handle both single results and arrays
                            if isinstance(file_results, list):
                                results.extend(file_results)
                            else:
                                results.append(file_results)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error reading {filename}: {e}, skipping...")
                        continue
            logger.info(f"Loaded {len(results)} benchmark results")
        return results

    async def save_benchmark(self, benchmark: BenchmarkResponse) -> Optional[str]:
        """Save benchmark results."""
        if self.use_mongo:
            try:
                result = await self.collection.insert_one(benchmark.dict())
                return str(result.inserted_id)
            except Exception as e:
                print(f"MongoDB save failed: {e}. Falling back to JSON.")
                self.use_mongo = False

        # JSON storage (default or fallback)
        try:
            # Create a new file for each benchmark with timestamp in name
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.json_dir, f"benchmark_{timestamp}.json")

            # Convert to dict and ensure prompt is included
            benchmark_dict = benchmark.dict()
            benchmark_dict['_id'] = timestamp
            benchmark_dict['timestamp'] = benchmark_dict['timestamp'].isoformat()

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(benchmark_dict, f, default=str, indent=2)

            return timestamp
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            return None

    async def get_benchmark_history(self, limit: int = 50) -> List[BenchmarkResponse]:
        """Retrieve benchmark history."""
        if self.use_mongo:
            try:
                cursor = self.collection.find().sort('timestamp', -1).limit(limit)
                results = await cursor.to_list(length=limit)
                return [BenchmarkResponse(**result) for result in results]
            except Exception as e:
                print(f"MongoDB query failed: {e}. Falling back to JSON.")
                self.use_mongo = False

        # JSON storage (default or fallback)
        try:
            # Load and merge results from all files
            results = self._load_existing_results()

            # Convert timestamps and sort
            results = [self._convert_timestamps(r) for r in results]
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            results = results[:limit]

            # Ensure each result has required fields
            valid_results = []
            for result in results:
                try:
                    # Ensure required fields exist
                    if 'prompt' not in result:
                        result['prompt'] = "Unknown prompt"

                    if 'system_info' not in result:
                        result['system_info'] = {
                            'platform': 'Unknown',
                            'processor': 'Unknown',
                            'python_version': 'Unknown',
                            'cpu': {'physical_cores': 0, 'total_cores': 0},
                            'memory': {'total': 0, 'available': 0, 'used': 0, 'percent_used': 0}
                        }

                    if 'results' not in result:
                        result['results'] = []

                    # Convert any string timestamps in results
                    if 'results' in result and isinstance(result['results'], list):
                        for r in result['results']:
                            if isinstance(r.get('timestamp'), str):
                                r['timestamp'] = datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00'))

                    valid_results.append(BenchmarkResponse(**result))
                except Exception as e:
                    print(f"Error converting result: {e}")
                    continue

            return valid_results
        except Exception as e:
            print(f"Error loading from JSON: {e}")
            return []

    async def get_benchmark_by_id(self, benchmark_id: str) -> Optional[BenchmarkResponse]:
        """Retrieve a specific benchmark by ID."""
        logger.info(f"Fetching benchmark with ID: {benchmark_id}")
        if self.use_mongo:
            try:
                result = await self.collection.find_one({'_id': benchmark_id})
                return BenchmarkResponse(**result) if result else None
            except Exception as e:
                logger.error(f"MongoDB query failed: {e}. Falling back to JSON.")
                self.use_mongo = False

        # JSON storage (default or fallback)
        try:
            # Search through all JSON files
            results = self._load_existing_results()
            for result in results:
                if result.get('_id') == benchmark_id:
                    # Convert timestamps before creating BenchmarkResponse
                    result = self._convert_timestamps(result)
                    logger.info(f"Found benchmark {benchmark_id}")
                    logger.debug(f"Benchmark data: {result}")
                    return BenchmarkResponse(**result)
            logger.warning(f"Benchmark {benchmark_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            return None