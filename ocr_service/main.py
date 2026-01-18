"""
Core API Gateway for the AL Financial OCR Service.
Entry point for running the application locally or via Mangum (AWS Lambda).
"""

from mangum import Mangum

from ocr_service.app import create_app

app = create_app()
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ocr_service.main:app", host="0.0.0.0", port=8000, reload=True)  # nosec B104
