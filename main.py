from PIL import ImageGrab
import cv2
import numpy as np

# from ultralytics import YOLO
from utils import *


def screen_capture() -> np.ndarray:
    img = ImageGrab.grab()
    frame = np.array(img)
    resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)


def main():
    while True:
        frame = screen_capture()
        result = detect_book(frame)
        print(result)
        print(result["predictions"][])


if __name__ == "__main__":
    main()
