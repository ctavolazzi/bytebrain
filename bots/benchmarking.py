import json
import logging
import time
from datetime import datetime
from statistics import mean, median
from .ollama_bot import OllamaBot

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    import os
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Setup logging with timestamp in filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/ollama_bot_{timestamp}.log'

    # Configure file logging only
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file)
        ]
    )
    return log_file

def analyze_chunks(chunks_data):
    """Analyze chunk patterns"""
    sizes = [len(chunk) for chunk in chunks_data]
    times = [t for t, _ in chunks_data[1:]]  # Skip first time as it's relative to start

    # Chunk size analysis
    avg_size = mean(sizes)
    med_size = median(sizes)
    max_size = max(sizes)
    min_size = min(sizes)

    # Time analysis
    intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
    avg_interval = mean(intervals) if intervals else 0
    max_interval = max(intervals) if intervals else 0

    # Detect bursts (chunks that come very quickly)
    burst_threshold = avg_interval / 2  # Define a burst as twice as fast as average
    burst_count = sum(1 for interval in intervals if interval < burst_threshold)

    return {
        'avg_chunk_size': avg_size,
        'median_chunk_size': med_size,
        'max_chunk_size': max_size,
        'min_chunk_size': min_size,
        'avg_interval': avg_interval,
        'max_interval': max_interval,
        'burst_chunks': burst_count
    }

def run_benchmark(config_path='config.json'):
    """Run benchmarking tests on models"""
    # Setup logging
    log_file = setup_logging()
    logging.info(f"Starting new test session. Logging to {log_file}")

    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            logging.info(f"Loaded config: {config}")
    except Exception as e:
        error_msg = f"Failed to load config: {e}"
        print(error_msg)
        logging.error(error_msg)
        return

    # Initialize bot
    try:
        bot = OllamaBot()
        logging.info("Successfully initialized OllamaBot")
    except Exception as e:
        error_msg = f"Failed to initialize OllamaBot: {e}"
        print(error_msg)
        logging.error(error_msg)
        return

    # Use default prompt from config
    prompt = config['default_prompt']
    logging.info(f"Using prompt: {prompt}")

    print("\n=== Starting Model Tests ===")
    print(f"Logging to: {log_file}\n")

    # Store results for comparison
    results = {}

    # Test each model
    for model in config['models']:
        print(f"\n=== Testing {model} ===")
        logging.info(f"Testing model: {model}")
        try:
            logging.info(f"Sending prompt to {model}: {prompt}")

            print(f"Prompt: {prompt}")
            print("Response: ", end='', flush=True)

            start_time = time.time()
            full_response = ""
            first_token_time = None
            chunk_count = 0
            chunks_data = []  # Store (timestamp, content) for each chunk

            for chunk in bot.generate_response(prompt, model=model, stream=True):
                chunk_count += 1
                current_time = time.time()
                if not first_token_time:
                    first_token_time = current_time

                content = chunk['message']['content']
                chunks_data.append((current_time - start_time, content))

                print(content, end='', flush=True)
                full_response += content

            end_time = time.time()

            # Basic metrics
            total_time = end_time - start_time
            time_to_first_token = first_token_time - start_time if first_token_time else 0
            word_count = len(full_response.split())
            chunks_per_second = chunk_count / total_time if total_time > 0 else 0
            words_per_second = word_count / total_time if total_time > 0 else 0

            # Advanced analysis
            chunk_analysis = analyze_chunks(chunks_data)

            print("\n")  # New line after response
            print(f"Time metrics:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Time to first token: {time_to_first_token:.2f}s")
            print(f"  Chunks: {chunk_count}")
            print(f"  Words: {word_count}")
            print(f"  Chunks per second: {chunks_per_second:.2f}")
            print(f"  Words per second: {words_per_second:.2f}")
            print("\nChunk analysis:")
            print(f"  Average chunk size: {chunk_analysis['avg_chunk_size']:.1f} chars")
            print(f"  Median chunk size: {chunk_analysis['median_chunk_size']:.1f} chars")
            print(f"  Chunk size range: {chunk_analysis['min_chunk_size']}-{chunk_analysis['max_chunk_size']} chars")
            print(f"  Average time between chunks: {chunk_analysis['avg_interval']*1000:.1f}ms")
            print(f"  Longest pause: {chunk_analysis['max_interval']*1000:.1f}ms")
            print(f"  Burst chunks: {chunk_analysis['burst_chunks']} ({(chunk_analysis['burst_chunks']/chunk_count*100):.1f}%)")
            print()

            # Store results
            results[model] = {
                'response': full_response,
                'total_time': total_time,
                'time_to_first_token': time_to_first_token,
                'chunk_count': chunk_count,
                'word_count': word_count,
                'chunks_per_second': chunks_per_second,
                'words_per_second': words_per_second,
                **chunk_analysis  # Include chunk analysis in results
            }

            # Log everything
            logging.info(f"Response from {model}: {full_response}")
            logging.info(f"Metrics for {model}: total_time={total_time:.2f}s, "
                        f"time_to_first_token={time_to_first_token:.2f}s, "
                        f"chunks={chunk_count}, words={word_count}, "
                        f"chunks_per_second={chunks_per_second:.2f}, "
                        f"words_per_second={words_per_second:.2f}")
            logging.info(f"Chunk analysis for {model}: {chunk_analysis}")

        except Exception as e:
            error_msg = f"Error with {model}: {e}"
            print(error_msg)
            logging.error(error_msg)

    # Print comparison
    if len(results) > 1:
        print("\n=== Model Comparison ===")
        metrics = [
            'total_time', 'time_to_first_token', 'chunk_count', 'word_count',
            'chunks_per_second', 'words_per_second', 'avg_chunk_size',
            'median_chunk_size', 'avg_interval', 'burst_chunks'
        ]
        for metric in metrics:
            print(f"\n{metric.replace('_', ' ').title()}:")
            for model, data in results.items():
                print(f"  {model}: {data[metric]:.2f}")

    logging.info("Test session completed")
    print("\n=== Test Session Completed ===")
    print(f"Full logs available in: {log_file}")

    return results
