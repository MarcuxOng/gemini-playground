from __future__ import annotations

import base64
import logging
import uuid

from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import build_genai_client, settings
from app.utils.gcs import get_gcs_bucket_name, upload_to_gcs

logger = logging.getLogger(__name__)
client = build_genai_client()


def _is_legacy_imagen(model: str) -> bool:
    """Check if the model uses the legacy Imagen generate_images API."""
    return model.startswith("imagen-")


def _image_bytes_to_url(data: bytes, mime_type: str = "image/png") -> str:
    """Return a base64 data URL for use in dev (no GCS dependency)."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _store_image(image_bytes: bytes, filepath: str, mime_type: str = "image/png") -> str:
    """Store generated image: GCS in production, base64 data URL in dev."""
    if get_gcs_bucket_name():
        return upload_to_gcs(image_bytes, filepath, mime_type)
    return _image_bytes_to_url(image_bytes, mime_type)


def _extract_imagen_bytes(response) -> list[tuple[bytes, str]]:
    """Extract image data from a legacy Imagen generate_images response."""
    images: list[tuple[bytes, str]] = []
    if not response or not response.generated_images:
        return images
    for generated_image in response.generated_images:
        if generated_image.image and generated_image.image.image_bytes:
            images.append((generated_image.image.image_bytes, "image/png"))
    return images


def _extract_nano_banana_bytes(response) -> list[tuple[bytes, str]]:
    """Extract inline image data from a Nano Banana generate_content response."""
    images: list[tuple[bytes, str]] = []
    if not response or not response.candidates:
        return images
    for candidate in response.candidates:
        if not candidate.content or not candidate.content.parts:
            continue
        for part in candidate.content.parts:
            if part.inline_data and part.inline_data.data:
                mime = part.inline_data.mime_type or "image/png"
                images.append((part.inline_data.data, mime))
    return images


def _generate_imagen(prompt: str, model: str) -> list[str]:
    """Generate images using the legacy Imagen generate_images API."""
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )
    urls: list[str] = []
    for i, (image_bytes, mime_type) in enumerate(_extract_imagen_bytes(response)):
        ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
        filename = f"{uuid.uuid4()}_{i}.{ext}"
        url = _store_image(image_bytes, f"image/{filename}", mime_type)
        urls.append(url)
    return urls


def _edit_imagen(prompt: str, base_image_bytes: bytes, model: str) -> list[str]:
    """Edit images using the legacy Imagen generate_images API."""
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
        ),
    )
    urls: list[str] = []
    for i, (image_bytes, mime_type) in enumerate(_extract_imagen_bytes(response)):
        ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
        filename = f"{uuid.uuid4()}_edit_{i}.{ext}"
        url = _store_image(image_bytes, f"image/{filename}", mime_type)
        urls.append(url)
    return urls


_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def generate_image_service(prompt: str, model: str = settings.gemini_image_model) -> list[str]:
    """Generates images from a text prompt.

    Routes to the legacy Imagen API for ``imagen-*`` models or the Nano Banana
    ``generate_content`` API for Gemini-native image models.
    """
    try:
        logger.info("Generating image with model '%s': %s", model, prompt)

        if _is_legacy_imagen(model):
            return _generate_imagen(prompt, model)

        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                safety_settings=_SAFETY_SETTINGS,
            ),
        )

        urls: list[str] = []
        for i, (image_bytes, mime_type) in enumerate(_extract_nano_banana_bytes(response)):
            ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
            filename = f"{uuid.uuid4()}_{i}.{ext}"
            url = _store_image(image_bytes, f"image/{filename}", mime_type)
            urls.append(url)

        return urls
    except Exception as e:
        logger.error("Error in generate_image_service: %s", e)
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def edit_image_service(
    prompt: str, base_image_bytes: bytes, model: str = settings.gemini_image_model
) -> list[str]:
    """Edits an image based on a text prompt.

    Routes to the legacy Imagen API for ``imagen-*`` models or the Nano Banana
    ``generate_content`` API for Gemini-native image models.
    """
    try:
        logger.info("Editing image with model '%s': %s", model, prompt)

        if _is_legacy_imagen(model):
            return _edit_imagen(prompt, base_image_bytes, model)

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=base_image_bytes, mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                safety_settings=_SAFETY_SETTINGS,
            ),
        )

        urls: list[str] = []
        for i, (image_bytes, mime_type) in enumerate(_extract_nano_banana_bytes(response)):
            ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
            filename = f"{uuid.uuid4()}_edit_{i}.{ext}"
            url = _store_image(image_bytes, f"image/{filename}", mime_type)
            urls.append(url)

        return urls
    except Exception as e:
        logger.error("Error in edit_image_service: %s", e)
        raise
