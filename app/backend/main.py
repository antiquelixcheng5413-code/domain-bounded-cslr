from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.routes import router

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "app/frontend"

app = FastAPI(
    title="Domain-Bounded CSL Recognition",
    version="0.1.0",
    description="Controlled research prototype. Not for medical decision-making.",
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=FRONTEND_ROOT), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")
