from PIL import ImageGrab
import cv2
import numpy as np
import time
from inference import *

# from ultralytics import YOLO
from utils import *


def screen_capture() -> np.ndarray:
    img = ImageGrab.grab()
    frame = np.array(img)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():

    time.sleep(3)
    frame = screen_capture()
    run(frame_to_base64(frame))


if __name__ == "__main__":
    main()
