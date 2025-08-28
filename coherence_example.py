import asyncio
import logging
import sys
import os

# Add the project's 'src' directory to the Python path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Now we can import the robust, tested components from the 'src' directory
from coherence_example import ask_zai, MockZAIClient

# =================================================
# Demonstration
#
# This script now serves as a simple entrypoint that uses the more
# complex, well-tested logic from the `src` directory.
# =================================================
async def main():
    """
    Runs a demonstration of the coherent response logic.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # The client from `src` is configurable. We'll simulate one incoherent response.
    client = MockZAIClient(api_key="DUMMY_API_KEY", incoherent_attempts=1)
    question = "Explique a diferença entre aprendizado supervisionado e não supervisionado."

    # We now use the ask_zai function from `src`, which has proper retry logic.
    # The function from `src` will handle the incoherent response and re-prompt automatically.
    final_answer = await ask_zai(question, client)

    print("\n=========================")
    print("=== Resposta Coerente ===")
    print("=========================\n")
    print(final_answer)

if __name__ == "__main__":
    try:
        # The `main` function from `src/coherence_example.py` uses a `finally`
        # block with `logging.shutdown()`. We will adopt that best practice here.
        asyncio.run(main())
    finally:
        logging.shutdown()
