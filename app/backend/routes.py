from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.backend.schemas import HealthResponse, PredictionResponse
from app.backend.services import create_recognition_service

MAX_UPLOAD_BYTES = int(os.getenv("CSLR_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
ALLOWED_SUFFIXES: set[str] = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
ALLOWED_CONTENT_TYPES: set[str] = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/x-matroska",
    "application/octet-stream",
}

router = APIRouter(prefix="/api/v1")
service = create_recognition_service()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=service.ready,
        demo_mode=service.demo_mode,
        model_error=service.model_error,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(video: Annotated[UploadFile, File()]) -> PredictionResponse:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        detail = f"unsupported video extension: {suffix or 'none'}"
        raise HTTPException(status_code=415, detail=detail)
    if video.content_type and video.content_type not in ALLOWED_CONTENT_TYPES:
        detail = f"unsupported content type: {video.content_type}"
        raise HTTPException(status_code=415, detail=detail)

    temporary_path = None
    size = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temporary_path = Path(handle.name)
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="video exceeds upload limit")
                handle.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="uploaded video is empty")
        result = service.predict_video(temporary_path)
        return PredictionResponse(**asdict(result))
    finally:
        await video.close()
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)
