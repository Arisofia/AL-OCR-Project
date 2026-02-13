"""Image utility functions."""
import base64
import logging
from typing import Optional, Union

logger = logging.getLogger("ocr-service.utils.image")


def decode_image(data: Union[str, bytes]) -> Optional[bytes]:
    """Decodes image from base64 string or returns bytes directly."""
    if isinstance(data, bytes):
        return data

    try:
        # Check if it's a data URL
        if data.startswith("data:image"):
            data = data.split(",")[1]
        return base64.b64decode(data)
    except (OSError, ValueError) as e:
        logger.error("Failed to decode image data: %s", e)
        return None


def load_image_from_path(path: str) -> Optional[bytes]:
    """Reads image bytes from local file path."""
    try:
        with open(path, "rb") as f:
            return f.read()
    except (OSError, ValueError) as e:
        logger.error("Failed to read image from path %s: %s", path, e)
        return None
