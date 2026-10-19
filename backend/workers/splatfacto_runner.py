"""
splatfacto_runner.py

Trains a Gaussian Splat with Nerfstudio's splatfacto and exports splat.ply.
Nerfstudio runs inside the Docker image immverse-nerfstudio:v1 (CUDA 12.4
toolchain), because the host's CUDA 13 / glibc 2.43 can't compile gsplat.
"""

import os
import shlex
import shutil
import subprocess
import time

NS_DOCKER_IMAGE = os.getenv("NS_DOCKER_IMAGE", "immverse-nerfstudio:v1")
CONTAINER_WORK = "/work"  # output_dir is always mounted here


class SplatfactoError(Exception):
    pass


def _run_in_docker(ns_cmd: list[str], step_name: str, log_file, host_dir: str):
    """Run one Nerfstudio command inside the container; stream output to the log."""
    uid, gid = os.getuid(), os.getgid()
    # chown at the end so files created by root in the container belong to us
    script = f"{shlex.join(ns_cmd)}; rc=$?; chown -R {uid}:{gid} {CONTAINER_WORK}; exit $rc"

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-e", "MAX_JOBS=2",
        "-e", "TORCH_CUDA_ARCH_LIST=7.5",
        "-v", f"{os.path.abspath(host_dir)}:{CONTAINER_WORK}",
        NS_DOCKER_IMAGE,
        "bash", "-c", script,
    ]

    log_file.write(f"\n=== {step_name} ===\n$ {' '.join(cmd)}\n")
    log_file.flush()

    result = subprocess.run(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,  # never wait on an interactive prompt
        text=True,
    )
    if result.returncode != 0:
        raise SplatfactoError(
            f"Step '{step_name}' failed with exit code {result.returncode}. "
            f"Check the log for details."
        )


def run_splatfacto(
    scan_id: str,
    images_dir: str,
    colmap_model_dir: str,
    output_dir: str,
    max_train_steps: int = 15000,
) -> str:
    """
    Trains a Gaussian Splat from a COLMAP reconstruction and exports it.
    Returns the host path to splat.ply. Raises SplatfactoError on failure.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "splatfacto.log")
    start = time.time()

    # Layout nerfstudio's colmap dataparser expects:
    #   <data_dir>/images/  and  <data_dir>/colmap/sparse/0
    data_dir = os.path.join(output_dir, "ns_data")
    ns_images_dir = os.path.join(data_dir, "images")
    ns_colmap_dir = os.path.join(data_dir, "colmap", "sparse", "0")
    os.makedirs(ns_images_dir, exist_ok=True)
    os.makedirs(ns_colmap_dir, exist_ok=True)

    for fname in os.listdir(images_dir):
        dst = os.path.join(ns_images_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(images_dir, fname), dst)

    for fname in os.listdir(colmap_model_dir):
        dst = os.path.join(ns_colmap_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(colmap_model_dir, fname), dst)

    experiment_name = f"scan_{scan_id}"

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n\n########## Splatfacto run for scan {scan_id} ##########\n")

        # Step 1: train (paths below are CONTAINER paths, under /work)
        _run_in_docker(
            [
                "ns-train", "splatfacto",
                "--data", f"{CONTAINER_WORK}/ns_data",
                "--output-dir", CONTAINER_WORK,
                "--experiment-name", experiment_name,
                "--max-num-iterations", str(max_train_steps),
                "--vis", "viewer_legacy",
                "--viewer.quit-on-train-completion", "True",
                "colmap",
                "--colmap-path", "colmap/sparse/0",
                "--images-path", "images",
                # Full-res frames: avoids the interactive "downscale? [y/n]" prompt
                "--downscale-factor", "1",
            ],
            "ns-train splatfacto",
            log,
            output_dir,
        )

        # Nerfstudio nests output: <output_dir>/<experiment>/splatfacto/<timestamp>/
        exp_dir = os.path.join(output_dir, experiment_name, "splatfacto")
        run_dirs = sorted(os.listdir(exp_dir)) if os.path.isdir(exp_dir) else []
        if not run_dirs:
            raise SplatfactoError(f"No training run found under {exp_dir}")
        latest = run_dirs[-1]
        container_config = f"{CONTAINER_WORK}/{experiment_name}/splatfacto/{latest}/config.yml"

        # Step 2: export splat.ply
        _run_in_docker(
            [
                "ns-export", "gaussian-splat",
                "--load-config", container_config,
                "--output-dir", f"{CONTAINER_WORK}/export",
            ],
            "ns-export gaussian-splat",
            log,
            output_dir,
        )

        log.write(f"\nSplatfacto finished in {time.time() - start:.1f} seconds\n")

    splat_file = os.path.join(output_dir, "export", "splat.ply")
    if not os.path.isfile(splat_file):
        raise SplatfactoError(
            f"Export ran but no splat.ply found at {splat_file}. Check splatfacto.log."
        )
    return splat_file


if __name__ == "__main__":
    # Manual test:
    #   python splatfacto_runner.py <images_dir> <colmap_model_dir> <output_dir> [steps]
    import sys

    if len(sys.argv) not in (4, 5):
        print("Usage: python splatfacto_runner.py <images_dir> <colmap_model_dir> <output_dir> [steps]")
        sys.exit(1)

    steps = int(sys.argv[4]) if len(sys.argv) == 5 else 15000
    print(f"Training splatfacto for {steps} steps (needs the GPU) ...")
    splat_path = run_splatfacto("manual-test", sys.argv[1], sys.argv[2], sys.argv[3], steps)
    print(f"Done. Splat file at: {splat_path}")