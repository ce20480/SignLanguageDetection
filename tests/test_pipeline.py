from asl.pipeline import ASLPipeline
from asl.detection.hand_detector import HandDetector
from asl.data.preprocessor import ASLPreprocessor
from asl.models.coords_model import CoordsModel
from asl.visualization.visualizer import ASLVisualizer
import cv2
import numpy as np
import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test full pipeline with a sample image
# Replace with paths to your model and sample image
model_path = "path/to/model.pt"
sample_image_path = "path/to/sample_hand_image.jpg"

if os.path.exists(model_path) and os.path.exists(sample_image_path):
    # Create pipeline components
    detector = HandDetector(min_detection_confidence=0.7)
    preprocessor = ASLPreprocessor(normalize=True, flatten=True)
    model = CoordsModel.load(model_path)
    visualizer = ASLVisualizer()
    
    # Create pipeline
    pipeline = ASLPipeline(detector, preprocessor, model, visualizer)
    
    # Load image
    image = cv2.imread(sample_image_path)
    
    # Process image through pipeline
    prediction, landmarks, processed_image = pipeline.process_image(image)
    
    if prediction is not None:
        print(f"Pipeline prediction: Class {prediction['class_index']} with confidence {prediction['confidence']:.4f}")
        
        # Display result
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: Class {prediction['class_index']} (Confidence: {prediction['confidence']:.4f})")
        plt.axis('off')
        plt.show()
        
        print("Pipeline integration test: PASSED")
    else:
        print("No hand detected in sample image.")
else:
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at {sample_image_path}")
