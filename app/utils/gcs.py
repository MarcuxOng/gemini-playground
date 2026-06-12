from __future__ import annotations

import logging
from datetime import timedelta

from google.cloud import storage
from google.cloud.storage import Bucket

from app.config import settings

logger = logging.getLogger(__name__)


def _get_bucket() -> Bucket | None:
    """Return the GCS bucket object, or None if not configured."""
    if not settings.gcs_bucket:
        logger.warning("GCS_BUCKET not configured")
        return None
    try:
        client = storage.Client()
        return client.bucket(settings.gcs_bucket)
    except Exception as e:
        logger.error(f"Failed to connect to GCS: {e}")
        return None


def upload_to_gcs(
    data: bytes, gcs_path: str, content_type: str = "application/octet-stream"
) -> str:
    """Upload data to GCS at the given path, return a signed URL or public URL."""
    bucket = _get_bucket()
    if not bucket:
        return f"https://placeholder.com/{gcs_path}"

    blob = bucket.blob(gcs_path)
    blob.upload_from_string(data, content_type=content_type)

    try:
        sign_kwargs: dict[str, object] = {"expiration": timedelta(hours=1)}
        if settings.gcp_service_account_email:
            sign_kwargs["service_account_email"] = settings.gcp_service_account_email
        url = blob.generate_signed_url(**sign_kwargs)
        return str(url)
    except Exception:
        logger.warning("Signed URL generation failed; falling back to public URL")
        return str(blob.public_url)


def delete_from_gcs(gcs_path: str) -> None:
    """Delete a blob from GCS. No-op if the bucket is not configured."""
    bucket = _get_bucket()
    if not bucket:
        return
    blob = bucket.blob(gcs_path)
    blob.delete()
    logger.info(f"Deleted GCS blob: {gcs_path}")


def gcs_uri(gcs_path: str) -> str:
    """Return the gs:// URI for a GCS blob path."""
    return f"gs://{settings.gcs_bucket}/{gcs_path}"
