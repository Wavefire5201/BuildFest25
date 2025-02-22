import cv2
import numpy as np
import pyautogui
import torch
from ultralytics import YOLO

model = YOLO()

# Screen resolution (adjust based on your setup)
screen_width, screen_height = pyautogui.size()


def capture_screen():
    """
    Captures the current screen and converts it to a NumPy array for OpenCV processing.
    """
    screenshot = pyautogui.screenshot()  # Capture screen
    frame = np.array(screenshot)  # Convert to NumPy array
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    return frame


def detect_book(frame):
    """
    Detects objects in the frame using the YOLOv5 model and checks for a fully opened book.
    """
    results = model(frame)  # Run detection
    detections = results.pandas().xyxy[0]  # Get detection results as a pandas DataFrame

    # Filter detections for "book" class (replace 'book' with your specific class label)
    books = detections[detections["name"] == "book"]

    for _, book in books.iterrows():
        x1, y1, x2, y2 = (
            int(book["xmin"]),
            int(book["ymin"]),
            int(book["xmax"]),
            int(book["ymax"]),
        )
        confidence = book["confidence"]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Book {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame


def main():
    """
    Main function to capture desktop stream and detect books in real-time.
    """
    while True:
        frame = capture_screen()  # Capture current screen
        frame_resized = cv2.resize(frame, (640, 480))  # Resize for faster processing

        processed_frame = detect_book(frame_resized)  # Detect books

        cv2.imshow(
            "Desktop Stream - Book Detection", processed_frame
        )  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit on pressing 'q'
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
