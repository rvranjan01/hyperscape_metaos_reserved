# Immverse Room Scan to 3D Gaussian Splat

Backend service for receiving room-scan frames, reconstructing a sparse model
with COLMAP, training a Gaussian Splat with Nerfstudio Splatfacto, aligning it
to optional headset poses, and uploading the resulting asset to Amazon S3.

The service is a FastAPI application backed by PostgreSQL. The API and worker
run on the host, while Splatfacto runs in the GPU-enabled Docker image
`immverse-nerfstudio:v1`. COLMAP is built with CUDA support on the host.

## Contents

- [Architecture and project layout](#architecture-and-project-layout)
- [Requirements](#requirements)
- [Local installation](#local-installation)
- [PostgreSQL with Docker](#postgresql-with-docker)
- [Configuration](#configuration)
- [Run the application](#run-the-application)
- [AWS EC2 GPU setup](#aws-ec2-gpu-setup)
- [CUDA and COLMAP installation](#cuda-and-colmap-installation)
- [Nerfstudio Docker image](#nerfstudio-docker-image)
- [Direct pipeline test](#direct-pipeline-test)
- [S3 verification](#s3-verification)
- [Training through the API](#training-through-the-api)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Architecture and project layout

```text
hyperscape_metaOS/
├── backend/
│   ├── main.py                 # FastAPI application and scan endpoints
│   ├── config.py               # .env configuration
│   ├── database.py             # SQLAlchemy models and PostgreSQL setup
│   ├── requirements.txt        # Python dependencies
│   ├── routes/
│   │   ├── upload.py           # Frame upload endpoint
│   │   └── metadata.py         # Headset metadata upload endpoint
│   └── workers/
│       ├── worker.py           # Polls for processing scans
│       ├── colmap_runner.py    # Host COLMAP pipeline
│       ├── splatfacto_runner.py # Docker/Nerfstudio pipeline
│       ├── align.py            # Optional pose alignment
│       └── s3_utils.py         # S3 upload and presigned URLs
├── uploads/                    # Temporary uploaded frames (created at runtime)
├── metadata/                   # Temporary scan metadata (created at runtime)
├── processing/                 # COLMAP and Splatfacto work files
└── .env                        # Local configuration; never commit this file
```

The worker processes one scan at a time:

```text
uploaded frames
    -> COLMAP feature extraction, matching, and mapping
    -> Splatfacto training in Docker
    -> optional pose alignment
    -> S3 upload
    -> completed scan and downloadable presigned URL
```

## Requirements

### Development machine

- Ubuntu, WSL2, or another Linux environment
- Python 3.12 (Python 3.11+ is supported by the project)
- Git
- Docker Engine or Docker Desktop with Linux containers
- PostgreSQL 14+ (local Docker PostgreSQL or Amazon RDS)
- AWS CLI configured with permissions for the configured S3 buckets

### AWS GPU host

- EC2 instance with an NVIDIA Tesla T4 or compatible CUDA GPU
- NVIDIA driver and Docker NVIDIA runtime
- At least 58 GB free disk space is recommended for the saved Nerfstudio
  image, build artifacts, and scan data
- AWS region `ap-south-1` in the commands below

## Local installation

Run these commands from the project directory:

```bash
cd ~/ranjan/hyperscape_metaOS

# Ubuntu/Debian prerequisites
sudo apt update
sudo apt install -y git curl build-essential python3.12 python3.12-venv python3.12-dev

# Create and activate the project virtual environment
python3.12 -m venv ~/ranjan/venv
source ~/ranjan/venv/bin/activate

# Install all backend dependencies
cd backend
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.12 is not available from the default Ubuntu repositories, install
it from the deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

The current dependency list includes FastAPI, Uvicorn, SQLAlchemy,
`psycopg2-binary`, multipart upload support, `python-dotenv`, Boto3, NumPy,
and HTTP client support. `DATABASE_URL` uses the PostgreSQL driver suffix
`+psycopg2`, for example `postgresql+psycopg2://...`.

## PostgreSQL with Docker

Use this option for local development when no PostgreSQL server is already
available:

```bash
docker volume create immverse-postgres-data

docker run -d \
  --name postgres-db \
  --restart unless-stopped \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=dev123 \
  -e POSTGRES_DB=immverse \
  -p 5432:5432 \
  -v immverse-postgres-data:/var/lib/postgresql/data \
  postgres:14

docker ps
docker logs postgres-db
```

On subsequent starts:

```bash
docker start postgres-db
```

Connect to PostgreSQL and inspect the tables:

```bash
docker exec -it postgres-db psql -U postgres -d immverse
```

Useful `psql` commands:

```sql
\dt
SELECT * FROM scans;
SELECT * FROM jobs;
SELECT * FROM assets;
\q
```

If PostgreSQL is running in RDS instead, do not start the local container.
Set `DATABASE_URL` to the RDS connection string and ensure the EC2 security
group allows the database port.

## Configuration

Create `~/ranjan/hyperscape_metaOS/.env` and replace every placeholder:

```env
DATABASE_URL=
AWS_REGION=ap-south-1
S3_BUCKET_RAW_CAPTURES=<raw-captures-bucket>
S3_BUCKET_SPLAT_OUTPUTS=<splat-outputs-bucket>
S3_BUCKET_APP_ASSETS=<app-assets-bucket>

# Required on the AWS host for the CUDA COLMAP build:
COLMAP_EXE=/home/ubuntu/colmap/build/src/colmap/exe/colmap

# Optional; these defaults are used when omitted:
NS_DOCKER_IMAGE=immverse-nerfstudio:v1
SPLAT_TRAIN_STEPS=15000
COLMAP_SEQUENTIAL_MIN_IMAGES=150
COLMAP_SEQUENTIAL_OVERLAP=20
```

The application loads `.env` from the project root, not from `backend/`.
AWS credentials should come from an EC2 IAM role or the AWS CLI credential
chain; do not place access keys in `.env` or commit them.

## Run the application

### Terminal 1: FastAPI server

```bash
cd ~/ranjan/hyperscape_metaOS/backend
source ~/ranjan/venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

For local-only development, `--reload` can be used:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: background worker

The worker polls PostgreSQL every 10 seconds for scans whose uploads are
complete:

```bash
cd ~/ranjan/hyperscape_metaOS/backend
source ~/ranjan/venv/bin/activate
python -m workers.worker
```

### Verify the server

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
```

Interactive documentation is available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`
- `http://127.0.0.1:8000/openapi.json`

## AWS EC2 GPU setup

The following procedure is for an EC2 GPU instance in `ap-south-1`.

### 1. Start and connect to the instance

1. In the AWS Console, select the instance and choose **Instance state >
   Start**.
2. Wait for both EC2 status checks to pass (`2/2 checks passed`).
3. Connect from Windows Command Prompt or VS Code Remote SSH:

```bash
ssh immverse-gpu
```

The `immverse-gpu` SSH alias should use AWS Systems Manager and the instance
ID so it continues to work when the public IP changes.

### 2. Verify the GPU and Docker installation

```bash
nvidia-smi
docker version
docker info | grep "Docker Root Dir"
docker images
```

The GPU must be visible to `nvidia-smi`. The image
`immverse-nerfstudio:v1` must be listed by `docker images`.

> **Persistence warning:** If `Docker Root Dir` points to an ephemeral NVMe
> volume such as `/opt/dlami/nvme`, Docker images may not survive an instance
> stop. Store Docker data on a persistent EBS-backed path when necessary.

### 3. Verify project files and S3 access

```bash
ls -la ~/ranjan/hyperscape_metaOS
ls -l ~/colmap/build/src/colmap/exe/colmap
aws sts get-caller-identity
aws s3 ls
```

## CUDA and COLMAP installation

These steps build the CUDA-enabled COLMAP executable used by
`COLMAP_EXE`. Run them on the EC2 host, not inside the Nerfstudio container.

### Install the CUDA toolkit

Use the CUDA repository version appropriate for the installed NVIDIA driver
and Ubuntu release. For Ubuntu 22.04 and CUDA 12.4, the repository-key method
is:

```bash
sudo apt update
sudo apt install -y wget gpg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Verify the toolkit:

```bash
/usr/local/cuda/bin/nvcc --version
nvidia-smi
```

The CUDA development packages can use approximately 4 GB of disk space.
Remove the downloaded keyring package if it is no longer needed:

```bash
rm -f cuda-keyring_1.1-1_all.deb
```

### Build COLMAP with CUDA support

Remove an older apt-installed COLMAP first to avoid binary conflicts:

```bash
sudo apt remove -y colmap
```

Install the compiler, build system, and COLMAP development dependencies:

```bash
sudo apt update
sudo apt install -y \
  git cmake ninja-build build-essential \
  libboost-program-options-dev libboost-graph-dev libboost-system-dev \
  libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
  libgoogle-glog-dev libgflags-dev libceres-dev libsqlite3-dev \
  libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev \
  libatlas-base-dev libsuitesparse-dev
```

Clone, configure, compile, and verify COLMAP:

```bash
cd ~
git clone https://github.com/colmap/colmap.git
cd ~/colmap
mkdir -p build
cd build

# Tesla T4 uses CUDA compute capability 7.5.
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=75
ninja -j"$(nproc)"

./src/colmap/exe/colmap -h
```

The resulting executable is:

```text
~/colmap/build/src/colmap/exe/colmap
```

Set the same path in `.env`:

```env
COLMAP_EXE=/home/ubuntu/colmap/build/src/colmap/exe/colmap
```

If a system-wide install is preferred, run this from the COLMAP build
directory:

```bash
sudo ninja install
which colmap
colmap -h
```

## Nerfstudio Docker image

Splatfacto is invoked automatically by `backend/workers/splatfacto_runner.py`
with:

- `--gpus all`
- `MAX_JOBS=2`
- `TORCH_CUDA_ARCH_LIST=7.5`
- the scan output mounted at `/work`

The host therefore needs the image before the worker processes a scan:

```bash
docker images immverse-nerfstudio:v1
docker run --rm --gpus all immverse-nerfstudio:v1 nvidia-smi
```

### Save the current container as an image

If the container already contains Nerfstudio, ffmpeg, CUDA-compatible
libraries, and the compiled `gsplat` cache, open a second terminal on the
host (outside the container) and run:

```bash
docker commit 7a447e0d4bf4 immverse-nerfstudio:v1
docker images immverse-nerfstudio:v1
```

`docker commit` snapshots the container filesystem as a reusable image. The
`MAX_JOBS` and `TORCH_CUDA_ARCH_LIST` environment variables are supplied by
the worker at runtime and are not saved by this command. Expect the image to
be approximately 12–15 GB. Confirm the Docker root directory is persistent
before stopping the EC2 instance.

## Direct pipeline test

This test bypasses the API and verifies COLMAP and Splatfacto independently.
Replace the input directory with a directory containing at least three
overlapping `.jpg`, `.jpeg`, or `.png` frames.

```bash
cd ~/ranjan/hyperscape_metaOS/backend
source ~/ranjan/venv/bin/activate

export COLMAP_EXE=~/colmap/build/src/colmap/exe/colmap
python workers/colmap_runner.py \
  ~/ranjan/test_images/horse \
  ~/ranjan/out_runner2
```

The COLMAP runner writes its sparse model under
`~/ranjan/out_runner2/sparse/`. Train and export a Splatfacto model using the
best sparse model:

```bash
python workers/splatfacto_runner.py \
  ~/ranjan/test_images/horse \
  ~/ranjan/out_runner2/sparse/0 \
  ~/ranjan/out_splat2 \
  500
```

Expected output:

```text
~/ranjan/out_splat2/export/splat.ply
```

The worker normally uses `15000` training steps. The `500` value above is
only a quick smoke test.

## S3 verification

Verify the configured bucket and upload a generated test asset:

```bash
cd ~/ranjan/hyperscape_metaOS/backend
source ~/ranjan/venv/bin/activate

python -c "from config import S3_BUCKET_SPLAT_OUTPUTS as B; from workers.s3_utils import upload_file; upload_file('/home/ubuntu/ranjan/out_splat2/export/splat.ply', B, 'scans/s3-smoke-test/splat/model.ply'); print('uploaded to', B)"

aws s3 ls s3://<splat-outputs-bucket>/scans/s3-smoke-test/ --recursive
```

Remove the smoke-test object after verification:

```bash
aws s3 rm s3://<splat-outputs-bucket>/scans/s3-smoke-test/splat/model.ply
```

Use the actual value of `S3_BUCKET_SPLAT_OUTPUTS` in place of
`<splat-outputs-bucket>`. The production worker stores results under:

```text
scans/<scan_id>/splat/model.ply
```

or, when pose alignment succeeds:

```text
scans/<scan_id>/splat/model_aligned.ply
```

## Training through the API

Start the FastAPI service in Terminal 1 and the worker in Terminal 2 as
described above. Use Terminal 3 for the client commands.

### 1. Create a scan

```bash
BASE_URL=http://127.0.0.1:8000

curl -sS -X POST "$BASE_URL/scans/create" \
  -H "Content-Type: application/json" \
  -d '{"device_name":"lighthouse-test","total_frames":20}'
```

Copy the returned `scan_id` into `SCAN_ID`:

```bash
SCAN_ID="<scan-id-from-response>"
```

### 2. Upload optional headset metadata

Metadata must be uploaded before the final image changes the scan status to
`processing`. The accepted filenames are:

- `cameras.txt`
- `poses.csv`
- `poses.json`
- `segments.json`
- `session_manifest.json`

The worker uses `cameras.txt` for COLMAP intrinsics and `poses.csv` for
alignment when those files are present.

```bash
curl -sS -X POST "$BASE_URL/scans/$SCAN_ID/metadata" \
  -F "file=@/path/to/cameras.txt"

curl -sS -X POST "$BASE_URL/scans/$SCAN_ID/metadata" \
  -F "file=@/path/to/poses.csv"
```

### 3. Upload image frames

```bash
cd ~/ranjan/hyperscape_metaOS

for img in ~/ranjan/test_images/lighthouse/*.jpg; do
  echo "Uploading: $img"
  curl -sS -X POST \
    "$BASE_URL/scans/$SCAN_ID/upload-chunk" \
    -F "chunk=@$img"
  echo
done
```

When the final expected frame arrives, the scan status changes from
`uploading` to `processing`. The worker then picks it up.

### 4. Poll status

```bash
curl -sS "$BASE_URL/scans/$SCAN_ID/status"
```

Poll until the response contains `"status":"complete"` or `"status":"failed"`:

```bash
while true; do
  curl -sS "$BASE_URL/scans/$SCAN_ID/status"
  echo
  sleep 10
done
```

Stop the loop with `Ctrl+C` after the terminal shows the desired state.

### 5. Retrieve the finished asset

```bash
curl -sS "$BASE_URL/scans/$SCAN_ID/asset"
```

For a completed scan, the response contains a presigned S3 URL valid for
3600 seconds.

## API reference

All endpoints are unauthenticated in the current implementation. The base
URL is `http://127.0.0.1:8000` for local execution or the EC2 host and port
when accessed remotely.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Returns the welcome message. |
| `GET` | `/health` | Returns the service status. |
| `POST` | `/scans/create` | Creates a scan record. |
| `GET` | `/scans` | Lists all scan records. |
| `GET` | `/scans/{scan_id}/status` | Returns upload and processing status. |
| `GET` | `/scans/{scan_id}/asset` | Returns a presigned URL after completion. |
| `POST` | `/scans/{scan_id}/upload-chunk` | Uploads one image/frame file as multipart field `chunk`. |
| `POST` | `/scans/{scan_id}/metadata` | Uploads one allowed metadata file as multipart field `file`. |

### `GET /`

```bash
curl -sS "$BASE_URL/"
```

Response:

```json
{"message":"Welcome to the Immverse Room Scan API!"}
```

### `GET /health`

```bash
curl -sS "$BASE_URL/health"
```

Response:

```json
{"status":"online","database":"connected"}
```

This endpoint currently reports the configured service state; it does not
perform a live database query.

### `POST /scans/create`

Request:

```json
{
  "device_name": "Quest 3s lighthouse",
  "total_frames": 309
}
```

`device_name` is required. `total_frames` defaults to `0`; use a positive
value for automatic transition to processing after all frames arrive.

Response:

```json
{
  "scan_id": "e4cb89c6-459c-4a43-8f79-effae2934650",
  "status": "uploading",
  "device_name": "Quest 3 lighthouse"
}
```

### `GET /scans`

```bash
curl -sS "$BASE_URL/scans"
```

Response:

```json
[
  {
    "scan_id": "e4cb89c6-459c-4a43-8f79-effae2934650",
    "status": "processing",
    "device_name": "Quest 3 lighthouse",
    "total_frames": 309
  }
]
```

### `POST /scans/{scan_id}/upload-chunk`

Upload one frame with the multipart field name `chunk`:

```bash
curl -sS -X POST "$BASE_URL/scans/$SCAN_ID/upload-chunk" \
  -F "chunk=@/path/to/00020.jpg"
```

Response:

```json
{
  "status": "received",
  "filename": "00020.jpg",
  "uploaded_frames": 20,
  "total_frames": 20,
  "scan_status": "processing"
}
```

The endpoint returns `404` when `scan_id` does not exist.

### `POST /scans/{scan_id}/metadata`

Upload one allowed metadata file with the multipart field name `file`:

```bash
curl -sS -X POST "$BASE_URL/scans/$SCAN_ID/metadata" \
  -F "file=@/path/to/session_manifest.json"
```

Response:

```json
{
  "status": "received",
  "filename": "session_manifest.json",
  "bytes": 1234
}
```

The endpoint returns `400` for unsupported filenames, `404` for an unknown
scan, `409` if the scan is no longer uploading, and `413` for files larger
than 5,000,000 bytes.

### `GET /scans/{scan_id}/status`

```bash
curl -sS "$BASE_URL/scans/$SCAN_ID/status"
```

Response:

```json
{
  "scan_id": "e4cb89c6-459c-4a43-8f79-effae2934650",
  "status": "processing",
  "uploaded_frames": 20,
  "total_frames": 20,
  "progress_percentage": 100.0
}
```

Possible statuses are `uploading`, `processing`, `complete`, and `failed`.
The endpoint returns `404` for an unknown scan.

### `GET /scans/{scan_id}/asset`

```bash
curl -sS "$BASE_URL/scans/$SCAN_ID/asset"
```

While processing:

```json
{
  "scan_id": "e4cb89c6-459c-4a43-8f79-effae2934650",
  "status": "processing",
  "message": "Not ready yet"
}
```

After successful processing:

```json
{
  "scan_id": "e4cb89c6-459c-4a43-8f79-effae2934650",
  "status": "complete",
  "download_url": "https://...",
  "expires_in": 3600
}
```

The endpoint returns `404` for an unknown scan or when no Splat asset has
been recorded.

## Troubleshooting

### `DATABASE_URL is missing`

Ensure `.env` exists at the project root:

```bash
ls -la ~/ranjan/hyperscape_metaOS/.env
grep -E '^(DATABASE_URL|AWS_REGION|S3_BUCKET_)' ~/ranjan/hyperscape_metaOS/.env
```

Do not print or share database passwords or AWS credentials.

### COLMAP cannot be found

```bash
ls -l ~/colmap/build/src/colmap/exe/colmap
export COLMAP_EXE=~/colmap/build/src/colmap/exe/colmap
```

Check the configured path in `.env` and restart the worker after changing it.

### Splatfacto fails in Docker

Check the image, GPU runtime, and available disk:

```bash
docker images immverse-nerfstudio:v1
docker run --rm --gpus all immverse-nerfstudio:v1 nvidia-smi
df -h
docker system df
```

The worker writes detailed logs under `processing/<scan_id>/`, including
`colmap.log` and `splatfacto.log`.

### A scan remains `processing`

Check the worker terminal and query the scan:

```bash
curl -sS "$BASE_URL/scans/$SCAN_ID/status"
find ~/ranjan/hyperscape_metaOS/processing/"$SCAN_ID" -maxdepth 3 -type f -print
```

The worker must remain running in a separate terminal. It also requires the
COLMAP executable, the Docker image, a visible GPU, valid S3 permissions, and
enough overlapping frames for COLMAP to register at least half of the input
images and produce at least 500 points.
