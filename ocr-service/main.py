import logging
import os
import time
import uuid
import io
from typing import List, Optional

import cv2
import numpy as np
import pytesseract
import boto3
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from mangum import Mangum

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr-service")

app = FastAPI(
    title="AL Financial OCR Service",
    description="Professional OCR service for financial documents and receipts",
    version="1.1.0"
)

# Configuration
API_KEY = os.getenv("OCR_API_KEY", "default_secret_key")
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
USE_TEXTRACT = os.getenv("USE_TEXTRACT", "false").lower() == "true"

s3_client = boto3.client('s3')
textract_client = boto3.client('textract')

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class OCRResult(BaseModel):
    filename: str
    text: str
    processing_time: float
    strategy_used: str
    s3_key: Optional[str] = None
    source: str # "Tesseract" or "Textract"

class HealthResponse(BaseModel):
    status: str
    tesseract_version: str
    aws_integration: bool

# Security
async def get_api_key(header_value: str = Security(api_key_header)):
    if header_value == API_KEY:
        return header_value
    raise HTTPException(
        status_code=403, 
        detail="Invalid or missing API Key"
    )

class OCRProcessor:
    @staticmethod
    def enhance_standard(img_gray):
        blurred = cv2.medianBlur(img_gray, 3)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def enhance_sharpened(img_gray):
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img_gray, -1, sharpen_kernel)
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @classmethod
    def process_tesseract(cls, image_bytes: bytes) -> tuple[str, str]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        strategies = [
            ("standard", lambda g: cls.enhance_standard(g)),
            ("sharpened", lambda g: cls.enhance_sharpened(g)),
            ("raw_gray", lambda g: g)
        ]
        best_text = ""
        best_strategy = "none"
        config = r'--oem 3 --psm 6 -l spa+eng'
        for name, func in strategies:
            processed = func(gray)
            text = pytesseract.image_to_string(processed, config=config).strip()
            if len(text) > len(best_text):
                best_text = text
                best_strategy = name
        return best_text, best_strategy

    @staticmethod
    def process_textract(image_bytes: bytes) -> str:
        try:
            response = textract_client.detect_document_text(
                Document={'Bytes': image_bytes}
            )
            text = ""
            for item in response["Blocks"]:
                if item["BlockType"] == "LINE":
                    text += item["Text"] + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Textract processing failed: {e}")
            raise e

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        version = pytesseract.get_tesseract_version()
        aws_check = bool(S3_BUCKET)
        return HealthResponse(status="healthy", tesseract_version=str(version), aws_integration=aws_check)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", tesseract_version="unknown", aws_integration=False)

@app.post("/ocr", response_model=OCRResult)
async def perform_ocr(
    file: UploadFile = File(...), 
    api_key: str = Depends(get_api_key)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    logger.info(f"Processing file: {file.filename}")
    start_time = time.time()
    
    try:
        contents = await file.read()
        
        if USE_TEXTRACT:
            text = OCRProcessor.process_textract(contents)
            strategy = "AWS_TEXTRACT_DIRECT"
            source = "Textract"
        else:
            text, strategy = OCRProcessor.process_tesseract(contents)
            source = "Tesseract"
        
        s3_key = None
        if S3_BUCKET:
            s3_key = f"documents/{uuid.uuid4()}-{file.filename}"
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=contents,
                ContentType=file.content_type,
                Metadata={
                    "original_filename": file.filename,
                    "ocr_strategy": strategy,
                    "ocr_source": source
                }
            )
            logger.info(f"File uploaded to S3: {s3_key}")

        processing_time = round(time.time() - start_time, 3)
        return OCRResult(
            filename=file.filename,
            text=text,
            processing_time=processing_time,
            strategy_used=strategy,
            s3_key=s3_key,
            source=source
        )
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Lambda handler
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
