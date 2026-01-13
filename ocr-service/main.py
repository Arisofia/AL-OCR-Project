import logging
import os
import time
import uuid
from typing import List, Optional

import boto3
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from mangum import Mangum

# Import our custom iterative modules
from modules.ocr_engine import IterativeOCREngine

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr-service")

app = FastAPI(
    title="AL Financial OCR Project",
    description="Professional Iterative OCR & Pixel Reconstruction Service",
    version="1.2.0"
)

# Configuration
API_KEY = os.getenv("OCR_API_KEY", "default_secret_key")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
s3_client = boto3.client('s3')

# Feature flag: reconstruction
ENABLE_RECONSTRUCTION = os.getenv("ENABLE_RECONSTRUCTION", "false").lower() in ("1", "true", "yes")
RECON_ITERATIONS = int(os.getenv("RECON_ITERATIONS", os.getenv("OCR_ITERATIONS", 3)))

# Initialize Engine
ocr_engine = IterativeOCREngine(iterations=int(os.getenv("OCR_ITERATIONS", 3)))

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class OCRIteration(BaseModel):
    iteration: int
    text_length: int
    preview_text: str

class OCRResponse(BaseModel):
    filename: str
    text: str
    iterations: List[OCRIteration]
    processing_time: float
    s3_key: Optional[str] = None
    reconstruction: Optional[dict] = None  # optional reconstruction metadata

# Security
async def get_api_key(header_value: str = Security(api_key_header)):
    if header_value == API_KEY:
        return header_value
    raise HTTPException(status_code=403, detail="Invalid API Key")

# Recon package availability (best-effort detection)
try:
    import ocr_reconstruct as _ocr_reconstruct_pkg  # type: ignore
    RECON_PKG_AVAILABLE = True
    RECON_PKG_VERSION = getattr(_ocr_reconstruct_pkg, "__version__", None)
except Exception:
    RECON_PKG_AVAILABLE = False
    RECON_PKG_VERSION = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/recon/status")
async def recon_status():
    """Return reconstruction availability and package metadata."""
    return {
        "reconstruction_enabled": ENABLE_RECONSTRUCTION,
        "package_installed": RECON_PKG_AVAILABLE,
        "package_version": RECON_PKG_VERSION
    }

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(
    file: UploadFile = File(...), 
    reconstruct: bool = False,
    api_key: str = Depends(get_api_key)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    contents = await file.read()
    
    try:
        # Process using the new Iterative Engine
        use_recon = reconstruct or ENABLE_RECONSTRUCTION
        result = ocr_engine.process_image(contents, use_reconstruction=use_recon)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        s3_key = None
        recon_s3_key = None
        if S3_BUCKET:
            s3_key = f"processed/{uuid.uuid4()}-{file.filename}"
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=contents,
                ContentType=file.content_type
            )
            # Store reconstructed image if present in result
            if result.get("reconstruction") and result["reconstruction"].get("meta"):
                # If the reconstruction produced an in-memory image, engine may include it under meta
                # For now we store the reconstruction preview text and meta only
                recon_s3_key = f"recon_meta/{uuid.uuid4()}-{file.filename}.json"
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=recon_s3_key,
                    Body=str(result["reconstruction"]).encode("utf-8"),
                    ContentType="application/json"
                )

        processing_time = round(time.time() - start_time, 3)
        return OCRResponse(
            filename=file.filename,
            text=result["text"],
            iterations=result["iterations"],
            processing_time=processing_time,
            s3_key=s3_key,
            reconstruction=result.get("reconstruction")
        )

    except Exception as e:
        logger.error(f"Failed to process {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error") from e

# Lambda Handler
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
