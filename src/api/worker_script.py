import os
import json
from redis import Redis

# sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# from core.model_interface import OCRBaseModel
# from src.model.tesseract_wrapper import TesseractOCR  # Example model import


redis_client = Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

# model = TesseractOCR()
# model.load("/path/to/weights")


def process_job(job_id):
    job_data = redis_client.get(f"job:{job_id}")
    if not job_data:
        return
    job = json.loads(job_data)
    # image = ... # Download or load image from job['url']
    # result = model.predict(image)
    result = {"text": "dummy result", "boxes": []}  # Placeholder
    job["status"] = "DONE"
    job["result"] = result
    redis_client.set(f"job:{job_id}", json.dumps(job))


def main():
    while True:
        # Use blocking pop to avoid high CPU usage when queue is empty
        task = redis_client.blpop("ocr_tasks", timeout=0)
        if task:
            _, job_id = task
            process_job(job_id.decode())


if __name__ == "__main__":
    main()
