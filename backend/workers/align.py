"""
align.py

Moves a trained Gaussian splat into a metric, upright frame using the
headset's own camera positions (poses.csv).

Frame chain: Nerfstudio frame -> COLMAP frame (undo dataparser_transforms.json)
             -> headset frame (similarity fit of camera centres, x negated so
             the left-handed Unity data becomes right-handed).

Output is right-handed, Y-up, metres. A Unity viewer must flip x back.
Known limitation: spherical-harmonic colour coefficients are not rotated, so
colours may shift slightly with viewing angle.
"""

import csv
import glob
import json
import os
import re
import subprocess
import tempfile

import numpy as np

MIN_MATCHED_FRAMES = 10
MAX_FIT_RMS_M = 0.1  # judgement call; the room test fit at 0.014 m
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class AlignmentError(Exception):
    pass


def _quat_to_mat(w, x, y, z):
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1) * 2
        return np.array([.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
        return np.array([(R[2,1]-R[1,2])/S, .25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S])
    if R[1,1] > R[2,2]:
        S = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
        return np.array([(R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, .25*S, (R[1,2]+R[2,1])/S])
    S = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
    return np.array([(R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, .25*S])


def _quat_mul(a, b):
    """a: one quaternion (4,), b: many (N,4); both (w, x, y, z)."""
    return np.stack([
        a[0]*b[:,0] - a[1]*b[:,1] - a[2]*b[:,2] - a[3]*b[:,3],
        a[0]*b[:,1] + a[1]*b[:,0] + a[2]*b[:,3] - a[3]*b[:,2],
        a[0]*b[:,2] - a[1]*b[:,3] + a[2]*b[:,0] + a[3]*b[:,1],
        a[0]*b[:,3] + a[1]*b[:,2] - a[2]*b[:,1] + a[3]*b[:,0],
    ], axis=1)


def _umeyama(src, dst):
    """Similarity transform with dst ~ s * R @ src + t."""
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / S.var(0).sum()
    return s, R, md - s * R @ ms


def _frame_number(name):
    m = re.search(r"(\d+)\.[A-Za-z]+$", name)
    return int(m.group(1)) if m else None


def align_splat(splat_ply, colmap_model_dir, poses_csv, train_output_dir, out_ply):
    """
    Writes an aligned copy of splat_ply to out_ply.
    Assumes the trailing number in each image name (frame_00001.jpg -> 1)
    equals frame_index in poses.csv.
    Returns {"scale_m_per_colmap_unit", "fit_rms_m", "matched_frames", "gaussians"}.
    Raises AlignmentError on any problem.
    """
    colmap_exe = os.getenv("COLMAP_EXE", "colmap")

    # 1. Nerfstudio's own re-framing:  ns = ns_scale * (Rn @ colmap + tn)
    found = sorted(
        glob.glob(os.path.join(train_output_dir, "**", "dataparser_transforms.json"),
                  recursive=True),
        key=os.path.getmtime,
    )
    if not found:
        raise AlignmentError(f"No dataparser_transforms.json under {train_output_dir}")
    with open(found[-1]) as f:
        tj = json.load(f)
    T = np.array(tj["transform"])
    Rn, tn, ns_scale = T[:, :3], T[:, 3], float(tj["scale"])

    # 2. COLMAP camera centres, via a temporary text export of the model
    col = {}
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [colmap_exe, "model_converter", "--input_path", colmap_model_dir,
             "--output_path", tmp, "--output_type", "TXT"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise AlignmentError(f"model_converter failed: {r.stderr[-300:]}")
        with open(os.path.join(tmp, "images.txt")) as f:
            for line in f:
                p = line.split()
                if line.startswith("#") or len(p) < 10 or not p[-1].lower().endswith(IMAGE_EXTS):
                    continue
                w, x, y, z, tx, ty, tz = map(float, p[1:8])
                n = _frame_number(p[-1])
                if n is not None:
                    col[n] = -_quat_to_mat(w, x, y, z).T @ np.array([tx, ty, tz])

    # 3. Headset positions, x negated (left-handed -> right-handed)
    hdr = {}
    with open(poses_csv) as f:
        for r in csv.DictReader(f):
            hdr[int(r["frame_index"])] = np.array(
                [-float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])])

    ids = sorted(set(col) & set(hdr))
    if len(ids) < MIN_MATCHED_FRAMES:
        raise AlignmentError(
            f"Only {len(ids)} frames match between COLMAP and poses.csv; "
            f"need {MIN_MATCHED_FRAMES}.")
    src = np.array([col[i] for i in ids])
    dst = np.array([hdr[i] for i in ids])
    s2, R2, t2 = _umeyama(src, dst)

    # 4. Combine into one transform from Nerfstudio frame to headset frame
    s_tot = s2 / ns_scale
    R_tot = R2 @ Rn.T
    t_tot = t2 - s2 * (R2 @ Rn.T @ tn)
    ns_cams = ns_scale * (src @ Rn.T + tn)
    rms = float(np.sqrt(((s_tot * (ns_cams @ R_tot.T) + t_tot - dst) ** 2).sum(1).mean()))
    if rms > MAX_FIT_RMS_M:
        raise AlignmentError(
            f"Camera fit error {rms:.3f} m is above {MAX_FIT_RMS_M} m; poses do not match the images.")

    # 5. Apply to every Gaussian
    with open(splat_ply, "rb") as f:
        raw = f.read()
    marker = b"end_header\n"
    hend = raw.index(marker) + len(marker)
    header, props, n = raw[:hend], [], 0
    for line in header.decode().splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[2])
        if line.startswith("property"):
            if line.split()[1] != "float":
                raise AlignmentError(f"Unexpected PLY property type: {line}")
            props.append(line.split()[2])
    d = np.frombuffer(raw[hend:], dtype=np.float32).reshape(n, len(props)).copy()
    ix = [props.index(k) for k in ("x", "y", "z")]
    isc = [props.index(f"scale_{i}") for i in range(3)]
    irt = [props.index(f"rot_{i}") for i in range(4)]
    d[:, ix] = s_tot * (d[:, ix] @ R_tot.T) + t_tot
    d[:, isc] += np.log(s_tot)
    q = _quat_mul(_mat_to_quat(R_tot), d[:, irt])
    d[:, irt] = q / np.linalg.norm(q, axis=1, keepdims=True)

    with open(out_ply, "wb") as f:
        f.write(header)
        f.write(d.astype(np.float32).tobytes())

    return {"scale_m_per_colmap_unit": float(s2), "fit_rms_m": rms,
            "matched_frames": len(ids), "gaussians": n}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6:
        print("Usage: python align.py <splat.ply> <colmap_model_dir> <poses.csv> "
              "<training_output_dir> <out.ply>")
        sys.exit(1)
    print(align_splat(*sys.argv[1:6]))
