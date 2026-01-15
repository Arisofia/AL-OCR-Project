from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
import uuid
import json
import os

app = FastAPI(title="Arisofia AL-OCR API")
redis_client = Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

class OCRRequest(BaseModel):
    image_url: str
    callback_url: str = None

@app.post("/v1/process")
async def process_document(request: OCRRequest):
    job_id = str(uuid.uuid4())
    task = {
        "id": job_id,
        "url": request.image_url,
        "status": "QUEUED"
    }
    redis_client.set(f"job:{job_id}", json.dumps(task))
    redis_client.rpush("ocr_tasks", job_id)
    return {"job_id": job_id, "status": "QUEUED", "message": "Processing started."}

@app.get("/v1/status/{job_id}")
async def get_status(job_id: str):
    data = redis_client.get(f"job:{job_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(data)
