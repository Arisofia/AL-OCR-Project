import base64
import json

import pytest

from ocr_service.worker import RedisWorker


class FakeRedis:
    def __init__(self):
        self._store = {}

    async def set(self, name, value, nx=False, _ex=None):
        # Simple SET NX
        if nx:
            if name in self._store:
                return None
            self._store[name] = value
            return True
        self._store[name] = value
        return True

    async def get(self, name):
        v = self._store.get(name)
        if v is None:
            return None
        return v.encode("utf-8") if isinstance(v, str) else v


class DummyEngine:
    def __init__(self):
        self.counter = 0

    async def process_image(self, _image_bytes, _use_reconstruction=False):
        self.counter += 1
        return {"text": "ok", "confidence": 1.0}


@pytest.mark.asyncio
async def test_worker_idempotency():
    redis = FakeRedis()
    worker = RedisWorker(redis_client=redis)  # type: ignore[arg-type]
    worker.engine = DummyEngine()  # type: ignore[assignment]

    job_id = "job-1"
    job_key = f"job:{job_id}"
    payload = {"id": job_id, "image_bytes": base64.b64encode(b"abc").decode("ascii")}
    await redis.set(job_key, json.dumps(payload))

    # First processing should run engine once
    await worker.process_job(job_id)
    assert getattr(worker.engine, "counter", 0) == 1

    # Second processing (same id) should be detected as duplicate and skipped
    await worker.process_job(job_id)
    assert getattr(worker.engine, "counter", 0) == 1

    # Check job status stored as COMPLETED
    stored = await redis.get(job_key)
    assert stored is not None
    data = json.loads(stored.decode("utf-8"))
    assert data["status"] == "COMPLETED"
