"""Idempotency behavior tests for Redis worker processing."""

import asyncio
import base64
import json
from typing import Any, cast

import fakeredis
import pytest

from ocr_service.config import Settings
from ocr_service.worker import RedisWorker


class DummyEngine:
    """Simple async OCR engine stub with call counter."""

    def __init__(self):
        self.counter = 0

    async def process_image(self, _image_bytes, _use_reconstruction=False):
        await asyncio.sleep(0)
        self.counter += 1
        return {"text": "ok", "confidence": 1.0}


@pytest.fixture
async def redis_client():
    client = fakeredis.FakeAsyncRedis()
    yield client
    await cast(Any, client).aclose()


@pytest.fixture
def dummy_settings():
    settings = Settings(ocr_api_key="test")
    settings.redis_idempotency_ttl = 3600
    return settings


@pytest.mark.asyncio
async def test_worker_idempotency_new(redis_client, dummy_settings):
    worker = RedisWorker(settings=dummy_settings, redis_client=redis_client)
    worker.engine = DummyEngine()

    job_id = "job-1"
    job_key = f"job:{job_id}"
    payload = {"id": job_id, "image_bytes": base64.b64encode(b"abc").decode("ascii")}
    await redis_client.set(job_key, json.dumps(payload))

    await worker.process_job(job_id)
    assert getattr(worker.engine, "counter", 0) == 1

    stored_job_data = json.loads((await redis_client.get(job_key)).decode("utf-8"))
    assert stored_job_data["status"] == "COMPLETED"
    assert stored_job_data["result"]["text"] == "ok"

    idempotency_status = json.loads(
        (await redis_client.get(f"idempotency:{job_id}")).decode("utf-8")
    )
    assert idempotency_status["status"] == "COMPLETED"
    assert idempotency_status["result"]["text"] == "ok"

    await worker.process_job(job_id)
    assert getattr(worker.engine, "counter", 0) == 1


@pytest.mark.asyncio
async def test_worker_idempotency_processing_state(redis_client, dummy_settings):
    worker = RedisWorker(settings=dummy_settings, redis_client=redis_client)
    worker.engine = DummyEngine()

    job_id = "job-2"
    job_key = f"job:{job_id}"
    payload = {"id": job_id, "image_bytes": base64.b64encode(b"def").decode("ascii")}
    await redis_client.set(job_key, json.dumps(payload))

    await redis_client.set(
        f"idempotency:{job_id}",
        json.dumps({"status": "PROCESSING", "request_id": "test-rid"}),
    )

    await worker.process_job(job_id)
    assert getattr(worker.engine, "counter", 0) == 0
    idempotency_status = json.loads(
        (await redis_client.get(f"idempotency:{job_id}")).decode("utf-8")
    )
    assert idempotency_status["status"] == "PROCESSING"


@pytest.mark.asyncio
async def test_worker_job_failure_resets_idempotency(redis_client, dummy_settings):
    worker = RedisWorker(settings=dummy_settings, redis_client=redis_client)

    class FailingEngine(DummyEngine):
        """OCR stub that always fails after one async turn."""

        async def process_image(self, _image_bytes, _use_reconstruction=False):
            await asyncio.sleep(0)
            self.counter += 1
            raise RuntimeError("Simulated OCR Failure")

    worker.engine = FailingEngine()

    job_id = "job-3"
    job_key = f"job:{job_id}"
    payload = {"id": job_id, "image_bytes": base64.b64encode(b"ghi").decode("ascii")}
    await redis_client.set(job_key, json.dumps(payload))

    await worker.process_job(job_id)
    assert getattr(worker.engine, "counter", 0) == 1

    stored_job_data = json.loads((await redis_client.get(job_key)).decode("utf-8"))
    assert stored_job_data["status"] == "FAILED"

    idempotency_status = json.loads(
        (await redis_client.get(f"idempotency:{job_id}")).decode("utf-8")
    )
    assert idempotency_status["status"] == "FAILED"


@pytest.mark.asyncio
async def test_worker_job_with_request_id(redis_client, dummy_settings):
    worker = RedisWorker(settings=dummy_settings, redis_client=redis_client)
    worker.engine = DummyEngine()

    job_id = "job-4"
    job_key = f"job:{job_id}"
    request_id = "test-req-id-123"
    payload = {
        "id": job_id,
        "image_bytes": base64.b64encode(b"jkl").decode("ascii"),
        "request_id": request_id,
    }
    await redis_client.set(job_key, json.dumps(payload))

    await worker.process_job(job_id)

    idempotency_status = json.loads(
        (await redis_client.get(f"idempotency:{job_id}")).decode("utf-8")
    )
    assert idempotency_status["request_id"] == request_id
    assert idempotency_status["status"] == "COMPLETED"

    stored_job_data = json.loads((await redis_client.get(job_key)).decode("utf-8"))
    assert stored_job_data["request_id"] == request_id
    assert stored_job_data["status"] == "COMPLETED"
