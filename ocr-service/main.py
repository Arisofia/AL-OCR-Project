import logging
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from mangum import Mangum

from config import get_settings, Settings
from schemas import OCRResponse, HealthResponse, ReconStatusResponse
from services.storage import StorageService
from modules.ocr_engine import IterativeOCREngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr-service")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.version
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency Providers
def get_storage_service(settings: Settings = Depends(get_settings)):
    return StorageService(bucket_name=settings.s3_bucket_name)

def get_ocr_engine(settings: Settings = Depends(get_settings)):
    return IterativeOCREngine(iterations=settings.ocr_iterations)

# Security
api_key_header = APIKeyHeader(name=settings.api_key_header_name, auto_error=False)

async def get_api_key(
    header_value: str = Security(api_key_header),
    settings: Settings = Depends(get_settings)
):
    if header_value == settings.ocr_api_key:
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", timestamp=time.time())

@app.get("/recon/status", response_model=ReconStatusResponse)
async def recon_status(settings: Settings = Depends(get_settings)):
    """Return reconstruction availability and package metadata."""
    return ReconStatusResponse(
        reconstruction_enabled=settings.enable_reconstruction,
        package_installed=RECON_PKG_AVAILABLE,
        package_version=RECON_PKG_VERSION
    )

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(
    file: UploadFile = File(...), 
    reconstruct: bool = False,
    advanced: bool = False,
    doc_type: str = "generic",
    api_key: str = Depends(get_api_key),
    settings: Settings = Depends(get_settings),
    storage_service: StorageService = Depends(get_storage_service),
    ocr_engine: IterativeOCREngine = Depends(get_ocr_engine)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    contents = await file.read()
    
    try:
        if advanced:
            # Use Advanced AI-driven reconstruction with continuous learning
            result = await ocr_engine.process_image_advanced(contents, doc_type=doc_type)
        else:
            # Process using the standard Iterative Engine
            use_recon = reconstruct or settings.enable_reconstruction
            result = ocr_engine.process_image(contents, use_reconstruction=use_recon)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        s3_key = storage_service.upload_file(
            content=contents,
            filename=file.filename,
            content_type=file.content_type
        )
        
        if result.get("reconstruction") and result["reconstruction"].get("meta"):
            storage_service.upload_json(
                data=result["reconstruction"],
                filename=file.filename
            )

        processing_time = round(time.time() - start_time, 3)
        return OCRResponse(
            filename=file.filename,
            text=result["text"],
            iterations=result.get("iterations", []),
            processing_time=processing_time,
            s3_key=s3_key,
            reconstruction=result.get("reconstruction")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

# Lambda Handler
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
