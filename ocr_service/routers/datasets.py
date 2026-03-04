"""Dataset upload router for model improvement workflows (protected)."""

import asyncio
import hashlib
import os
import re
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ocr_service.routers.deps import (
    get_api_key,
    get_dataset_upload_key,
    get_storage_service,
)
from ocr_service.services.storage import StorageService
from ocr_service.utils.limiter import limiter

router = APIRouter(prefix="/datasets")


def _safe_name(value: str) -> str:
    value = os.path.basename(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or "upload.bin"


@router.post(
    "/upload",
    responses={
        400: {"description": "Invalid image upload"},
        500: {"description": "Dataset upload/storage error"},
    },
)
@limiter.limit("30/minute")
async def upload_dataset_image(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    dataset: Annotated[str, Form("occlusion_cards")],
    split: Annotated[str, Form("inbox")],
    doc_type: Annotated[str, Form("bank_card")],
    occlusion_type: Annotated[str, Form("unknown")],
    use_reconstruction: Annotated[bool, Form(False)],
    notes: Annotated[str, Form("")],
    _api_key: Annotated[str, Depends(get_api_key)],
    _dataset_key: Annotated[str, Depends(get_dataset_upload_key)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
) -> dict:
    """
    Upload a dataset image into S3 under a datasets/ prefix with sidecar metadata.

    This endpoint is protected by X-API-KEY and X-DATASET-KEY.
    """
    # SlowAPI's limiter requires `request` in the endpoint signature.
    _ = request

    if not storage.bucket_name:
        raise HTTPException(status_code=500, detail="S3 bucket not configured")

    content_type = (file.content_type or "application/octet-stream").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    safe_filename = _safe_name(file.filename or "upload.bin")
    sha256 = hashlib.sha256(raw).hexdigest()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    unique = uuid.uuid4().hex[:10]

    # Keep dataset keys predictable and scoped.
    dataset_key = re.sub(r"[^A-Za-z0-9_-]+", "_", (dataset or "unknown").strip())
    split_key = re.sub(r"[^A-Za-z0-9_-]+", "_", (split or "inbox").strip())
    base_prefix = f"datasets/{dataset_key}/{split_key}"
    obj_key = f"{base_prefix}/{stamp}_{sha256[:12]}_{unique}-{safe_filename}"
    meta_key = f"{base_prefix}/meta/{stamp}_{sha256[:12]}_{unique}.json"

    ok = await asyncio.to_thread(storage.put_object, obj_key, raw, content_type)
    if not ok:
        raise HTTPException(status_code=500, detail="Upload failed")

    meta = {
        "dataset": dataset_key,
        "split": split_key,
        "doc_type": doc_type,
        "occlusion_type": occlusion_type,
        "use_reconstruction": use_reconstruction,
        "notes": notes,
        "file_name": safe_filename,
        "content_type": content_type,
        "size_bytes": len(raw),
        "sha256": sha256,
        "s3_key": obj_key,
        "uploaded_at": time.time(),
    }
    await asyncio.to_thread(storage.save_json, meta, meta_key)

    return {
        "dataset": dataset_key,
        "split": split_key,
        "s3_key": obj_key,
        "meta_key": meta_key,
        "sha256": sha256,
        "size_bytes": len(raw),
    }
