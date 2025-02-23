import base64
from io import BytesIO
import os
import cv2
import numpy as np


def image_to_base64(image_path):
    """Converts an image file to a Base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def frame_to_base64(frame: np.ndarray):
    """
    Encodes a captured frame into base64 string.
    """
    # Encode the frame as JPEG
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Could not encode image")

    # Convert to base64
    base64_image = base64.b64encode(encoded_image).decode("utf-8")
    return base64_image
