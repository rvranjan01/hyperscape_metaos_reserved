import os

from database import Scan, SessionLocal
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

METADATA_DIR = "./metadata"  
MAX_BYTES = 5_000_000        # judgement call; these files are small text
# Only these names are accepted, so a client can't choose a path.
ALLOWED_FILES = {
    "cameras.txt",
    "poses.csv",
    "poses.json",
    "segments.json",
    "session_manifest.json",
}


@router.post("/scans/{scan_id}/metadata")
async def upload_metadata(scan_id: str, file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if name not in ALLOWED_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"File name must be one of: {sorted(ALLOWED_FILES)}",
        )

    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    db = SessionLocal()
    try:
        scan = db.query(Scan).filter(Scan.id == scan_id).first()
        if not scan:
            raise HTTPException(status_code=404, detail="Scan ID not found")
        # Once the last image arrives the worker may already have started,
        # so late metadata would be silently ignored. Refuse it instead.
        if scan.status != "uploading":
            raise HTTPException(
                status_code=409,
                detail=f"Scan is '{scan.status}'; send metadata before the last image.",
            )
    finally:
        db.close()

    scan_dir = os.path.join(METADATA_DIR, scan_id)
    os.makedirs(scan_dir, exist_ok=True)
    with open(os.path.join(scan_dir, name), "wb") as f:
        f.write(data)

    return {"status": "received", "filename": name, "bytes": len(data)}
