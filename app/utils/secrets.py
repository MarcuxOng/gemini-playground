from __future__ import annotations

import logging
import os

from google.cloud import secretmanager

logger = logging.getLogger(__name__)


def get_secret(secret_id: str, version_id: str = "latest") -> str | None:
    """Fetches a secret from Google Cloud Secret Manager."""
    try:
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            logger.error("GCP_PROJECT_ID not set in environment.")
            return None

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        return payload
    except Exception as e:
        logger.error(f"Error fetching secret {secret_id}: {e}")
        return None
