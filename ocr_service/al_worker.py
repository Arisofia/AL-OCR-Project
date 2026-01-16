"""
Standalone worker to trigger the Active Learning cycle.
Can be run via cron job or manually.
"""

import asyncio
import logging
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
from ocr_service.modules.active_learning_orchestrator import (  # noqa: E402
    ALOrchestrator,
)
from ocr_service.modules.learning_engine import LearningEngine  # noqa: E402
from ocr_service.utils.custom_logging import setup_logging  # noqa: E402


async def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("al-worker")

    logger.info("Initializing Active Learning Worker...")
    engine = LearningEngine()
    orchestrator = ALOrchestrator(engine)

    try:
        result = await orchestrator.run_cycle(n_samples=20)
        logger.info("Cycle completed: %s", result)
    except Exception as e:
        logger.exception("Worker failed during execution: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
