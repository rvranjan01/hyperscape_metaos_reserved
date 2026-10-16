# """
# colmap_runner.py

# Wraps the COLMAP CLI to turn a folder of frames into a sparse 3D reconstruction.

# Pipeline: feature_extractor -> exhaustive_matcher -> mapper
# Output: a "sparse/0" folder containing cameras.bin, images.bin, points3D.bin
#         (this is what Nerfstudio/Splatfacto reads in Week 2 step 2)

# Windows note: COLMAP must be installed and on PATH, OR set COLMAP_EXE below to
# the full path of COLMAP.bat / colmap.exe from the Windows release zip.
# Download: https://github.com/colmap/colmap/releases (grab the Windows build)
# """

# import os
# import subprocess
# import time

# # If "colmap" isn't on PATH, point this at your COLMAP.bat, e.g.:
# # Check properly this line when running in AWS
# # COLMAP_EXE = r"C:\colmap-x64-windows-nocuda\COLMAP.bat"

# # Path to the COLMAP binary. Override with the COLMAP_EXE environment variable
# # (e.g. in .env on AWS). Falls back to "colmap" on PATH.
# COLMAP_EXE = os.getenv("COLMAP_EXE", "colmap")


# class ColmapError(Exception):
#     pass


# def _run(cmd: list[str], step_name: str, log_file):
#     """Run a subprocess command, streaming output to a log file. Raises on failure."""
#     log_file.write(f"\n=== {step_name} ===\n$ {' '.join(cmd)}\n")
#     log_file.flush()

#     result = subprocess.run(
#         cmd,
#         stdout=log_file,
#         stderr=subprocess.STDOUT,
#         text=True,
#     )

#     if result.returncode != 0:
#         raise ColmapError(
#             f"COLMAP step '{step_name}' failed with exit code {result.returncode}. "
#             f"Check the log for details."
#         )


# def run_colmap(scan_id: str, images_dir: str, output_dir: str) -> str:
#     """
#     Runs the full COLMAP sparse reconstruction pipeline on a folder of images.

#     Args:
#         scan_id: the scan's UUID (used for logging only)
#         images_dir: folder containing frame_0000.jpg, frame_0001.jpg, ...
#         output_dir: folder to write colmap outputs into
#                     (a "database.db" file and a "sparse/" folder will be created here)

#     Returns:
#         path to the sparse model folder (output_dir/sparse/0), which is what
#         splatfacto_runner.py needs next.

#     Raises:
#         ColmapError if any COLMAP step fails.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     db_path = os.path.join(output_dir, "database.db")
#     sparse_dir = os.path.join(output_dir, "sparse")
#     os.makedirs(sparse_dir, exist_ok=True)

#     log_path = os.path.join(output_dir, "colmap.log")
#     start = time.time()

#     with open(log_path, "a") as log:
#         log.write(f"\n\n########## COLMAP run for scan {scan_id} ##########\n")

#         # Step 1: extract SIFT features from every image
#         _run(
#             [
#                 COLMAP_EXE, "feature_extractor",
#                 "--database_path", db_path,
#                 "--image_path", images_dir,
#                 "--ImageReader.single_camera", "1",  # all frames from same headset camera
#             ],
#             "feature_extractor",
#             log,
#         )

#         # Step 2: match features across all image pairs
#         # (exhaustive_matcher is correct for <~500 images; if you have more,
#         #  switch to sequential_matcher since Quest frames are captured in order)
#         _run(
#             [
#                 COLMAP_EXE, "exhaustive_matcher",
#                 "--database_path", db_path,
#             ],
#             "exhaustive_matcher",
#             log,
#         )

#         # Step 3: incremental structure-from-motion -> sparse point cloud + camera poses
#         _run(
#             [
#                 COLMAP_EXE, "mapper",
#                 "--database_path", db_path,
#                 "--image_path", images_dir,
#                 "--output_path", sparse_dir,
#             ],
#             "mapper",
#             log,
#         )

#         elapsed = time.time() - start
#         log.write(f"\nCOLMAP finished in {elapsed:.1f} seconds\n")

#     model_dir = os.path.join(sparse_dir, "0")
#     if not os.path.isdir(model_dir):
#         raise ColmapError(
#             f"COLMAP ran but produced no model at {model_dir}. "
#             "This usually means not enough image overlap/features were found — "
#             "check colmap.log and try scanning with more overlap between frames."
#         )

#     return model_dir


# if __name__ == "__main__":
#     # Quick manual test:
#     #   python colmap_runner.py <path_to_images_folder> <path_to_output_folder>
#     import sys

#     if len(sys.argv) != 3:
#         print("Usage: python colmap_runner.py <images_dir> <output_dir>")
#         sys.exit(1)

#     images_dir, output_dir = sys.argv[1], sys.argv[2]
#     print(f"Running COLMAP on {images_dir} -> {output_dir} ...")
#     model_path = run_colmap("manual-test", images_dir, output_dir)
#     print(f"Done. Sparse model at: {model_path}")


"""
colmap_runner.py

Wraps the COLMAP CLI to turn a folder of frames into a sparse 3D reconstruction.

Pipeline: feature_extractor -> matcher -> mapper -> pick best model
Output: a sparse model folder (cameras.bin, images.bin, points3D.bin) that
        splatfacto_runner.py trains on.

COLMAP_EXE comes from the environment (.env on AWS); falls back to "colmap" on PATH.
"""

import os
import re
import shutil
import subprocess
import time

COLMAP_EXE = os.getenv("COLMAP_EXE", "colmap")

# Above this many images, use the sequential matcher (frames are captured in
# order) instead of exhaustive matching, which compares every pair of images.
# 150 is a judgement call, not a measured threshold.
SEQUENTIAL_MIN_IMAGES = int(os.getenv("COLMAP_SEQUENTIAL_MIN_IMAGES", "150"))
SEQUENTIAL_OVERLAP = int(os.getenv("COLMAP_SEQUENTIAL_OVERLAP", "20"))

# A model smaller than this counts as a failed reconstruction. These two
# numbers are also judgement calls: adjust them if they reject good scans.
MIN_REGISTERED_FRACTION = 0.5
MIN_POINTS = 500

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class ColmapError(Exception):
    pass


def _run(cmd: list[str], step_name: str, log_file):
    """Run a subprocess command, streaming output to a log file. Raises on failure."""
    log_file.write(f"\n=== {step_name} ===\n$ {' '.join(cmd)}\n")
    log_file.flush()

    result = subprocess.run(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        raise ColmapError(
            f"COLMAP step '{step_name}' failed with exit code {result.returncode}. "
            f"Check the log for details."
        )


def _read_intrinsics(path: str):
    """Read the first camera from a COLMAP-format cameras.txt.
    Line format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
    Returns (model, "p1,p2,...")."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                return parts[1], ",".join(parts[4:])
    raise ColmapError(f"No camera found in {path}")


def _analyze_model(model_dir: str):
    """Returns (registered_images, points) for a sparse model folder."""
    result = subprocess.run(
        [COLMAP_EXE, "model_analyzer", "--path", model_dir],
        capture_output=True,
        text=True,
    )
    text = result.stdout + result.stderr
    reg = re.search(r"Registered images:\s*(\d+)", text)
    pts = re.search(r"Points:\s*(\d+)", text)
    return (int(reg.group(1)) if reg else 0, int(pts.group(1)) if pts else 0)


def run_colmap(scan_id: str, images_dir: str, output_dir: str,
               intrinsics_file: str | None = None) -> str:
    """
    Runs the COLMAP sparse reconstruction on a folder of images.

    Args:
        scan_id: scan UUID (logging only)
        images_dir: folder containing the frames (names should sort in capture
            order, e.g. frame_00001.jpg, for the sequential matcher)
        output_dir: folder for database.db, sparse/ and colmap.log
        intrinsics_file: optional COLMAP-format cameras.txt from the headset

    Returns:
        path to the best sparse model folder (the one with the most registered images).

    Raises:
        ColmapError if a step fails or the reconstruction is too small.
    """
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")

    # Clean start: a leftover database makes COLMAP skip images it has already
    # processed, which hides problems on re-runs.
    if os.path.exists(db_path):
        os.remove(db_path)
    shutil.rmtree(sparse_dir, ignore_errors=True)
    os.makedirs(sparse_dir)

    n_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS)])
    if n_images < 3:
        raise ColmapError(f"Only {n_images} images found in {images_dir}; need at least 3.")

    extract_cmd = [
        COLMAP_EXE, "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",  # all frames from the same headset camera
    ]
    if intrinsics_file:
        cam_model, cam_params = _read_intrinsics(intrinsics_file)
        extract_cmd += [
            "--ImageReader.camera_model", cam_model,
            "--ImageReader.camera_params", cam_params,
        ]

    if n_images > SEQUENTIAL_MIN_IMAGES:
        matcher_name = "sequential_matcher"
        match_cmd = [
            COLMAP_EXE, "sequential_matcher",
            "--database_path", db_path,
            "--SequentialMatching.overlap", str(SEQUENTIAL_OVERLAP),
        ]
    else:
        matcher_name = "exhaustive_matcher"
        match_cmd = [COLMAP_EXE, "exhaustive_matcher", "--database_path", db_path]

    log_path = os.path.join(output_dir, "colmap.log")
    start = time.time()

    with open(log_path, "a") as log:
        log.write(f"\n\n########## COLMAP run for scan {scan_id} "
                  f"({n_images} images, {matcher_name}) ##########\n")

        _run(extract_cmd, "feature_extractor", log)
        _run(match_cmd, matcher_name, log)
        _run(
            [
                COLMAP_EXE, "mapper",
                "--database_path", db_path,
                "--image_path", images_dir,
                "--output_path", sparse_dir,
            ],
            "mapper",
            log,
        )

        # COLMAP may split a scene into several models (0, 1, ...). Pick the biggest.
        models = sorted(
            d for d in os.listdir(sparse_dir)
            if os.path.isdir(os.path.join(sparse_dir, d))
        )
        if not models:
            raise ColmapError(
                "COLMAP produced no model. Usually not enough overlap or features "
                "between frames; check colmap.log."
            )

        best = None
        for d in models:
            path = os.path.join(sparse_dir, d)
            reg, pts = _analyze_model(path)
            log.write(f"\nmodel {d}: {reg} registered images, {pts} points\n")
            if best is None or reg > best[1]:
                best = (path, reg, pts)

        path, reg, pts = best
        if reg < MIN_REGISTERED_FRACTION * n_images or pts < MIN_POINTS:
            raise ColmapError(
                f"COLMAP reconstruction too small: {reg}/{n_images} images registered, "
                f"{pts} points. Capture needs more overlap or camera movement."
            )

        log.write(f"\nCOLMAP finished in {time.time() - start:.1f} seconds\n")

    return path


if __name__ == "__main__":
    # Manual test:
    #   python colmap_runner.py <images_dir> <output_dir> [cameras.txt]
    import sys

    if len(sys.argv) not in (3, 4):
        print("Usage: python colmap_runner.py <images_dir> <output_dir> [cameras.txt]")
        sys.exit(1)

    intr = sys.argv[3] if len(sys.argv) == 4 else None
    print(f"Running COLMAP on {sys.argv[1]} -> {sys.argv[2]} ...")
    model_path = run_colmap("manual-test", sys.argv[1], sys.argv[2], intr)
    print(f"Done. Sparse model at: {model_path}")