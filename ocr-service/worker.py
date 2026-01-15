import redis
import json

r = redis.Redis(host="localhost", port=6379, db=0)


def process_queue():
    while True:
        _, job_id = r.blpop("ocr_queue")
        job_id = job_id.decode("utf-8")
        data = json.loads(r.get(f"job:{job_id}"))
        data["status"] = "PROCESSING"
        r.set(f"job:{job_id}", json.dumps(data))
        # --- HEAVY OCR INFERENCE HERE ---
        result = {"text": "Sample Extracted Text"}
        # --------------------------------
        data["status"] = "COMPLETED"
        data["result"] = result
        r.set(f"job:{job_id}", json.dumps(data))


if __name__ == "__main__":
    process_queue()
