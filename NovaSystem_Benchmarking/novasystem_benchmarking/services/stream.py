import asyncio

# Global queue for streaming updates
stream_queue = asyncio.Queue()

async def send_update(update: dict):
    """Send an update to the stream queue."""
    await stream_queue.put(update)