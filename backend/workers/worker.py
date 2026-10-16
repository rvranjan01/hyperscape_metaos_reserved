# """
# worker.py

# Background worker that polls the database for scans marked "processing" and
# runs them through: COLMAP -> Splatfacto -> mark complete.

# MVP design (per plan): simple DB polling, one job at a time. No SQS, no
# concurrency. Run this as a second terminal/process alongside `uvicorn main:app`.

# Usage:
#     cd backend
#     venv\\Scripts\\activate
#     python -m workers.worker
# """

# import os
# import sys
# import time

# # allow `python -m workers.worker` to import sibling modules (database.py etc.)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from database import Scan, Job, Asset, SessionLocal
# from config import S3_BUCKET_SPLAT_OUTPUTS
# from workers.colmap_runner import run_colmap, ColmapError
# from workers.splatfacto_runner import run_splatfacto, SplatfactoError
# from workers.s3_utils import upload_file

# UPLOAD_DIR = "./uploads"       # matches routes/upload.py
# PROCESSING_DIR = "./processing"  # colmap + splat working directory
# POLL_INTERVAL_SECONDS = 10


# def find_next_scan_to_process(db):
#     """Grab the oldest scan whose uploads are done but not yet processed."""
#     return (
#         db.query(Scan)
#         .filter(Scan.status == "processing")
#         .order_by(Scan.created_at.asc())
#         .first()
#     )


# def process_scan(scan_id: str):
#     db = SessionLocal()
#     scan = db.query(Scan).filter(Scan.id == scan_id).first()
#     if not scan:
#         db.close()
#         return

#     images_dir = os.path.join(UPLOAD_DIR, scan_id)
#     work_dir = os.path.join(PROCESSING_DIR, scan_id)
#     colmap_out = os.path.join(work_dir, "colmap")
#     splat_out = os.path.join(work_dir, "splat")

#     # Track the pipeline stage as a Job row so /status can report it later
#     job = Job(scan_id=scan_id, stage="colmap", status="running")
#     db.add(job)
#     db.commit()

#     try:
#         print(f"[worker] scan {scan_id}: running COLMAP on {images_dir}")
#         model_dir = run_colmap(scan_id, images_dir, colmap_out)

#         job.stage = "splat_training"
#         db.commit()

#         print(f"[worker] scan {scan_id}: running Splatfacto")
#         splat_file = run_splatfacto(scan_id, images_dir, model_dir, splat_out)
        
#         # Upload the finished model to S3 so it survives even if this
#         # instance is reclaimed, and so the Quest app can download it.
#         s3_key = f"scans/{scan_id}/splat/model.ply"
#         print(f"[worker] scan {scan_id}: uploading to s3://{S3_BUCKET_SPLAT_OUTPUTS}/{s3_key}")
#         upload_file(splat_file, S3_BUCKET_SPLAT_OUTPUTS, s3_key)


#         # Record the output asset so GET /scans/{id}/asset can find it
#         asset = Asset(scan_id=scan_id, file_type="splat", s3_key=s3_key)
#         db.add(asset)

#         job.stage = "complete"
#         job.status = "done"
#         scan.status = "complete"
#         db.commit()
#         print(f"[worker] scan {scan_id}: COMPLETE -> {splat_file}")

#     except Exception as e:
#         job.status = "failed"
#         scan.status = "failed"
#         db.commit()
#         print(f"[worker] scan {scan_id}: FAILED - {e}")

#     finally:
#         db.close()


# def main_loop():
#     print("[worker] started. Polling for scans to process every "
#           f"{POLL_INTERVAL_SECONDS}s. Ctrl+C to stop.")
#     os.makedirs(PROCESSING_DIR, exist_ok=True)

#     while True:
#         db = SessionLocal()
#         scan = find_next_scan_to_process(db)
#         scan_id = scan.id if scan else None
#         db.close()

#         if scan_id:
#             process_scan(scan_id)
#         else:
#             time.sleep(POLL_INTERVAL_SECONDS)


# if __name__ == "__main__":
#     main_loop()


"""
worker.py

Background worker that polls the database for scans marked "processing" and
runs them through: COLMAP -> Splatfacto -> align -> upload to S3 -> complete.

MVP design: simple DB polling, one job at a time. No SQS, no concurrency.
Run this as a second process alongside `uvicorn main:app`.

Usage (from the backend folder):
    source ~/ranjan/venv/bin/activate
    python -m workers.worker
"""

import os
import sys
import time
import traceback

# allow `python -m workers.worker` to import sibling modules (database.py etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Scan, Job, Asset, SessionLocal
from config import S3_BUCKET_SPLAT_OUTPUTS
from workers.colmap_runner import run_colmap
from workers.splatfacto_runner import run_splatfacto
from workers.align import align_splat
from workers.s3_utils import upload_file

UPLOAD_DIR = "./uploads"         # matches routes/upload.py
METADATA_DIR = "./metadata"      # matches routes/metadata.py
PROCESSING_DIR = "./processing"  # colmap + splat working directory
POLL_INTERVAL_SECONDS = 10
TRAIN_STEPS = int(os.getenv("SPLAT_TRAIN_STEPS", "15000"))


def find_next_scan_to_process(db):
    """Grab the oldest scan whose uploads are done but not yet processed."""
    return (
        db.query(Scan)
        .filter(Scan.status == "processing")
        .order_by(Scan.created_at.asc())
        .first()
    )


def process_scan(scan_id: str):
    db = SessionLocal()
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        db.close()
        return

    images_dir = os.path.join(UPLOAD_DIR, scan_id)
    work_dir = os.path.join(PROCESSING_DIR, scan_id)
    colmap_out = os.path.join(work_dir, "colmap")
    splat_out = os.path.join(work_dir, "splat")

    # Optional headset metadata, stored by POST /scans/{id}/metadata
    meta_dir = os.path.join(METADATA_DIR, scan_id)
    cameras_txt = os.path.join(meta_dir, "cameras.txt")
    poses_csv = os.path.join(meta_dir, "poses.csv")
    intrinsics = cameras_txt if os.path.isfile(cameras_txt) else None
    poses = poses_csv if os.path.isfile(poses_csv) else None

    # Track the pipeline stage as a Job row so /status can report it later
    job = Job(scan_id=scan_id, stage="colmap", status="running")
    db.add(job)
    db.commit()

    try:
        print(f"[worker] scan {scan_id}: running COLMAP on {images_dir} "
              f"(intrinsics: {'yes' if intrinsics else 'no'})")
        model_dir = run_colmap(scan_id, images_dir, colmap_out, intrinsics)

        job.stage = "splat_training"
        db.commit()

        print(f"[worker] scan {scan_id}: running Splatfacto ({TRAIN_STEPS} steps)")
        splat_file = run_splatfacto(scan_id, images_dir, model_dir, splat_out,
                                    max_train_steps=TRAIN_STEPS)

        # Align to the headset frame (metres, Y-up). Optional: any failure here
        # falls back to the raw splat so the scan still completes.
        upload_path, s3_name = splat_file, "model.ply"
        if poses:
            job.stage = "aligning"
            db.commit()
            aligned_path = os.path.join(os.path.dirname(splat_file), "splat_aligned.ply")
            try:
                info = align_splat(splat_file, model_dir, poses, splat_out, aligned_path)
                print(f"[worker] scan {scan_id}: aligned, fit error "
                      f"{info['fit_rms_m']:.3f} m, {info['scale_m_per_colmap_unit']:.4f} m/unit")
                upload_path, s3_name = aligned_path, "model_aligned.ply"
            except Exception as e:
                print(f"[worker] scan {scan_id}: ALIGNMENT FAILED ({e}); uploading raw splat")
        else:
            print(f"[worker] scan {scan_id}: no poses.csv; uploading raw (unaligned) splat")

        # Upload the finished model to S3 so it survives even if this
        # instance is reclaimed, and so the Quest app can download it.
        s3_key = f"scans/{scan_id}/splat/{s3_name}"
        print(f"[worker] scan {scan_id}: uploading to s3://{S3_BUCKET_SPLAT_OUTPUTS}/{s3_key}")
        upload_file(upload_path, S3_BUCKET_SPLAT_OUTPUTS, s3_key)

        # Record the output asset so GET /scans/{id}/asset can find it
        asset = Asset(scan_id=scan_id, file_type="splat", s3_key=s3_key)
        db.add(asset)

        job.stage = "complete"
        job.status = "done"
        scan.status = "complete"
        db.commit()
        print(f"[worker] scan {scan_id}: COMPLETE -> {upload_path}")

    except Exception as e:
        job.status = "failed"
        scan.status = "failed"
        db.commit()
        print(f"[worker] scan {scan_id}: FAILED - {e}")
        traceback.print_exc()

    finally:
        db.close()


def main_loop():
    print("[worker] started. Polling for scans to process every "
          f"{POLL_INTERVAL_SECONDS}s. Ctrl+C to stop.")
    os.makedirs(PROCESSING_DIR, exist_ok=True)

    while True:
        db = SessionLocal()
        scan = find_next_scan_to_process(db)
        scan_id = scan.id if scan else None
        db.close()

        if scan_id:
            process_scan(scan_id)
        else:
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()