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
        results = detect_book(frame)
        print(results)
        for result in results["predictions"]:
            print(result)
            if result["class"] == "book" and result["confidence"] > 80:
                print("book detected!")


if __name__ == "__main__":
    main()
