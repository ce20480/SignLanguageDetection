import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from asl.detection.hand_detector import HandDetector
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize detector
detector = HandDetector()

# Test webcam capture (run this interactively)
def test_webcam_detection():
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Cannot open webcam")
    #     return
    
    # ret, frame = cap.read()
    # if not ret:
    #     print("Cannot read frame")
    #     return
    
    # # Test detection on a single frame
    # landmarks = detector.detect_from_image(frame)
    
    # if landmarks is not None:
    #     print(f"Hand detection successful. Found {len(landmarks)} landmarks.")
        
    #     # Visualize frame with landmarks
    #     frame_copy = frame.copy()
    #     for i, (x, y, _) in enumerate(landmarks):
    #         px, py = int(x * frame.shape[1]), int(y * frame.shape[0])
    #         cv2.circle(frame_copy, (px, py), 5, (0, 255, 0), -1)
            
    #     plt.figure(figsize=(10, 8))
    #     plt.imshow(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
    #     plt.title("Detected Hand Landmarks")
    #     plt.axis('off')
    #     plt.show()
    # else:
    #     print("No hand detected in webcam frame.")
    
    # cap.release()
    detector.detect_from_webcam(display=True)

# Call this function when ready to test webcam
test_webcam_detection()
