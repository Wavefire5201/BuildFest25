import base64
import cv2
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="uFBTEoar7nTQNxfCY9yk"
)


def image_to_base64(image_path):
    """Converts an image file to a Base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def frame_to_base64(frame):
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


def detect_book(frame):
    return CLIENT.infer(frame_to_base64(frame), model_id="coco/34")
