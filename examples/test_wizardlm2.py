"""Simple script to test WizardLM2 model."""

import asyncio
from nova_system.core import NovaSystemCore

async def main():
    # Initialize the system
    system = NovaSystemCore()

    try:
        # Test message
        print("Sending test message to WizardLM2...")
        result = await system.process_message(
            "Hello! Can you introduce yourself and tell me what makes you special?"
        )

        if result["success"]:
            print("\nResponse received:")
            print("-" * 50)
            print(result["response"])
            print("-" * 50)
            print("\nProcessing time:", result["processing_time_seconds"], "seconds")
        else:
            print("\nError:", result["error"])

    except Exception as e:
        print(f"Error: {e}")
    finally:
        system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())