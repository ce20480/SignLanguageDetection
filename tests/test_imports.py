import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Test main package imports
    from asl.data.dataset import ASLDataset
    from asl.data.preprocessor import ASLPreprocessor
    from asl.models.coords_model import CoordsModel
    from asl.models.model_factory import ModelFactory
    from asl.detection.hand_detector import HandDetector
    from asl.visualization.visualizer import ASLVisualizer
    from asl.pipeline import ASLPipeline
    
    # Test dependency imports
    import numpy as np
    import cv2
    import torch
    import mediapipe as mp
    import matplotlib.pyplot as plt
    
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
