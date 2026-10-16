"""
s3_utils.py

Thin wrapper around boto3 for uploading/downloading files to S3.

On the EC2 instance, this uses the IAM role attached to the instance
automatically -- no access keys needed. If running locally on Windows
without an IAM role, boto3 will look for credentials via `aws configure`
instead (ask your tech lead whether local access keys are provided, or
just test this file's functions once you're on the EC2 box).
"""

import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

from config import AWS_REGION

# Pin the regional endpoint and SigV4 signing. Without this, presigned links
# point at the global s3.amazonaws.com host and S3 answers with a
# TemporaryRedirect instead of the file.
_s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    endpoint_url=f"https://s3.{AWS_REGION}.amazonaws.com",
    config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
)


def upload_file(local_path: str, bucket: str, s3_key: str) -> str:
    """
    Uploads a local file to S3. Returns the s3:// URI on success.
    Raises ClientError if the upload fails (missing permissions, bad bucket, etc).
    """
    _s3.upload_file(local_path, bucket, s3_key)
    return f"s3://{bucket}/{s3_key}"


def download_file(bucket: str, s3_key: str, local_path: str) -> str:
    """Downloads a file from S3 to a local path. Returns the local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _s3.download_file(bucket, s3_key, local_path)
    return local_path


def generate_presigned_url(bucket: str, s3_key: str, expires_in: int = 3600) -> str:
    """
    Generates a temporary download link for a private S3 file.
    Default expiry: 1 hour. Use this for the /scans/{id}/asset endpoint
    so the Quest app can download the finished splat without needing
    AWS credentials itself.
    """
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=expires_in,
    )


def check_bucket_access(bucket: str) -> bool:
    """Quick test: can this instance/credentials reach the given bucket?"""
    try:
        _s3.head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        print(f"Cannot access bucket {bucket}: {e}")
        return False


if __name__ == "__main__":
    # Quick manual test: python s3_utils.py
    from config import S3_BUCKET_RAW_CAPTURES, S3_BUCKET_SPLAT_OUTPUTS, S3_BUCKET_APP_ASSETS

    for name, bucket in [
        ("raw-captures", S3_BUCKET_RAW_CAPTURES),
        ("splat-outputs", S3_BUCKET_SPLAT_OUTPUTS),
        ("app-assets", S3_BUCKET_APP_ASSETS),
    ]:
        ok = check_bucket_access(bucket)
        print(f"{name}: {'OK' if ok else 'FAILED'} ({bucket})")