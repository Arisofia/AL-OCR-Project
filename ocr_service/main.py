"""
Core API Gateway for the AL Financial OCR Service.
Entry point for running the application locally or via Mangum (AWS Lambda).
"""

import os

from mangum import Mangum

from ocr_service.app import create_app

app = create_app()
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    try:
        DEV_PORT = int(os.getenv("OCR_DEV_PORT", "8000"))
    except ValueError:
        DEV_PORT = 8000

    DEV_HOST = os.getenv("OCR_DEV_HOST", "127.0.0.1")

    uvicorn.run(
        "ocr_service.main:app",
        host=DEV_HOST,
        port=DEV_PORT,
        reload=True,
    )
