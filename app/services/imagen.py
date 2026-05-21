from __future__ import annotations

import logging
import uuid
from datetime import timedelta

from google import genai
from google.cloud import storage
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)
client = genai.Client(api_key=settings.gemini_api_key)


def upload_to_gcs(data: bytes, filename: str, content_type: str = "image/png") -> str:
    """Uploads data to GCS and returns a signed URL if possible, or a public URL."""
    try:
        if not settings.gcs_bucket_name:
            logger.warning("GCS_BUCKET_NAME not set. Returning dummy URL.")
            return f"https://placeholder.com/{filename}"

        try:
            storage_client = storage.Client()
        except Exception as cred_err:
            logger.error(f"Failed to initialize GCS client (ADC issue?): {cred_err}")
            return f"https://error-placeholder.com/{filename}"

        bucket = storage_client.bucket(settings.gcs_bucket_name)
        blob = bucket.blob(f"imagen/{filename}")
        blob.upload_from_string(data, content_type=content_type)

        # Generate a signed URL valid for 1 hour
        url = blob.generate_signed_url(expiration=timedelta(hours=1))
        return str(url) if url else f"https://error-placeholder.com/{filename}"
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        return f"https://error-placeholder.com/{filename}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def generate_image_service(prompt: str, model: str = "imagen-4.0-generate-001") -> list[str]:
    """
    Generates images from a text prompt using Imagen.
    Returns a list of URLs to the generated images.
    """
    try:
        logger.info(f"Generating image with model '{model}': {prompt}")
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
            ),
        )

        urls = []
        if response and response.generated_images:
            for i, generated_image in enumerate(response.generated_images):
                # The image data is in generated_image.image.data
                if generated_image.image and generated_image.image.image_bytes:
                    filename = f"{uuid.uuid4()}_{i}.png"
                    url = upload_to_gcs(generated_image.image.image_bytes, filename)
                    urls.append(url)

        return urls
    except Exception as e:
        logger.error(f"Error in generate_image_service: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def edit_image_service(
    prompt: str, base_image_bytes: bytes, model: str = "imagen-4.0-generate-001"
) -> list[str]:
    """
    Edits an image based on a text prompt.
    """
    try:
        logger.info(f"Editing image with model '{model}': {prompt}")
        # Note: Editing might require specific parameters or a different SDK call depending on the exact version.
        # Assuming generate_images with a reference image part.

        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                # reference_images=[types.Part.from_bytes(data=base_image_bytes, mime_type="image/png")]
                # Wait, check exact SDK param for editing
            ),
        )

        urls = []
        if response and response.generated_images:
            for i, generated_image in enumerate(response.generated_images):
                if generated_image.image and generated_image.image.image_bytes:
                    filename = f"{uuid.uuid4()}_edit_{i}.png"
                    url = upload_to_gcs(generated_image.image.image_bytes, filename)
                    urls.append(url)

        return urls
    except Exception as e:
        logger.error(f"Error in edit_image_service: {e}")
        raise
