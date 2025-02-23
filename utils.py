import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageGrab


def capture_desktop_frame() -> str:
    """
    Captures the current desktop frame and returns it as a Base64-encoded string.

    Returns:
        str: Base64-encoded screenshot of the desktop.
    """
    screenshot = ImageGrab.grab()
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_to_base64(image_path: str) -> str:
    """
    Converts an image file to a Base64-encoded string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def frame_to_base64(frame: np.ndarray) -> str:
    """
    Encodes a captured frame into a Base64-encoded string.

    Args:
        frame (np.ndarray): The image frame to encode.

    Returns:
        str: Base64-encoded string of the frame.

    Raises:
        ValueError: If the image cannot be encoded.
    """
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Could not encode image")
    return base64.b64encode(encoded_image).decode("utf-8")
