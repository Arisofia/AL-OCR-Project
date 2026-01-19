"""
Standalone worker to trigger the Active Learning cycle.
Can be run via cron job or manually.
"""

import asyncio
import logging
import sys

from ocr_service.config import get_settings
from ocr_service.modules.active_learning_mocks import MockOCRModel
from ocr_service.modules.active_learning_orchestrator import (
    ALOrchestrator,
)
from ocr_service.modules.learning_engine import LearningEngine
from ocr_service.utils.monitoring import init_monitoring


async def main():
    settings = get_settings()
    init_monitoring(settings)
    logger = logging.getLogger("al-worker")

    logger.info("Initializing Active Learning Worker...")
    engine = LearningEngine()
    mock_model = MockOCRModel()
    orchestrator = ALOrchestrator(engine, model=mock_model)

    try:
        result = await orchestrator.run_cycle(n_samples=20)
        logger.info("Cycle completed: %s", result)
    except Exception as e:
        logger.exception("Worker failed during execution: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
