"""
api/routes/uploads.py

POST /uploads — the saved file's own real path becomes the data_source
directly. No "upload:<id>" prefix scheme needed: load_data() already
accepts any real .csv/.parquet path as-is (built during the data-
generalization work), so nothing extra is required to bridge upload
storage into ProblemSpec.data_source.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from api.schemas_api import UploadResponse

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS ={ ".csv", ".parquet"}

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

router = APIRouter(prefix="/uploads", tags=["uploads"])

@router.post("", response_model=UploadResponse)
async def upload_dataset(file: UploadFile) -> UploadResponse:
    # file.filename can be None (some clients omit it) — Path(None) raises
    # TypeError, which would surface as an opaque 500 instead of a 400.
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        # The received filename is echoed back deliberately. The extension
        # alone is useless for debugging: an empty '' tells you nothing about
        # WHY it was empty, and the most common cause is a client that sent
        # the file without its name attached (e.g. PowerShell's `curl`, which
        # is an alias for Invoke-WebRequest and does not understand
        # -F "file=@path"), not a genuinely wrong file type.
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type {extension!r} "
                f"(received filename: {filename!r}). "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}. "
                "If the filename above is empty or has no extension, the "
                "upload client did not send the file's name — use real "
                "curl.exe, or the /docs page, rather than PowerShell's "
                "curl alias."
            ),
        )

    # Random filename, not the user's own — blocks path traversal via a
    # crafted filename, and prevents two uploads from colliding.
    stored_path = UPLOAD_DIR / f"{uuid.uuid4()}{extension}"

    size = 0
    with open(stored_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                stored_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (max 50MB).")
            f.write(chunk)

    return UploadResponse(file_path=str(stored_path), filename=file.filename, size_bytes=size)