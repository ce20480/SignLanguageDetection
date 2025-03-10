from asl.visualization.visualizer import ASLVisualizer
from asl.detection.hand_detector import HandDetector
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create visualizer
visualizer = ASLVisualizer()

# Test with sample image and landmarks
# Replace with a path to a sample image that has a hand
sample_image_path = "path/to/sample_hand_image.jpg"

if os.path.exists(sample_image_path):
    # Load image
    image = cv2.imread(sample_image_path)
    
    # Detect landmarks
    detector = HandDetector()
    landmarks = detector.detect_from_image(image)
    
    if landmarks is not None:
        # Test landmark plotting
        fig = visualizer.plot_landmarks(landmarks)
        plt.show()
        print("Landmark visualization: PASSED")
        
        # Test prediction visualization
        dummy_prediction = {
            'class_index': 0,
            'confidence': 0.95
        }
        
        result_image = visualizer.visualize_prediction(image, landmarks, dummy_prediction)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title("Prediction Visualization")
        plt.axis('off')
        plt.show()
        print("Prediction visualization: PASSED")
        
        # Test training history plotting
        dummy_history = {
            'train_loss': [0.9, 0.8, 0.7, 0.6, 0.5],
            'train_acc': [0.6, 0.7, 0.8, 0.85, 0.9],
            'val_loss': [0.95, 0.85, 0.75, 0.65, 0.55],
            'val_acc': [0.55, 0.65, 0.75, 0.8, 0.85]
        }
        
        history_fig = visualizer.plot_training_history(dummy_history)
        plt.show()
        print("Training history visualization: PASSED")
    else:
        print("No hand detected in sample image.")
else:
    print(f"Sample image not found at {sample_image_path}")
