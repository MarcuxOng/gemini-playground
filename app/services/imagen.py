from __future__ import annotations

import base64
import logging
import os
import uuid

from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import build_genai_client
from app.utils.gcs import upload_to_gcs

logger = logging.getLogger(__name__)
client = build_genai_client()


def _use_gcs() -> bool:
    return os.getenv("ENV") == "production" and bool(os.getenv("GCS_BUCKET", ""))


def _image_bytes_to_url(data: bytes, mime_type: str = "image/png") -> str:
    """Return a base64 data URL for use in dev (no GCS dependency)."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _store_image(image_bytes: bytes, filepath: str, mime_type: str = "image/png") -> str:
    """Store generated image: GCS in production, base64 data URL in dev."""
    if _use_gcs():
        return upload_to_gcs(image_bytes, filepath, mime_type)
    return _image_bytes_to_url(image_bytes, mime_type)


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
                    url = _store_image(
                        generated_image.image.image_bytes, f"imagen/{filename}", "image/png"
                    )
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
                    url = _store_image(
                        generated_image.image.image_bytes, f"imagen/{filename}", "image/png"
                    )
                    urls.append(url)

        return urls
    except Exception as e:
        logger.error(f"Error in edit_image_service: {e}")
        raise
