from asl.pipeline import ASLPipeline
from asl.detection.hand_detector import HandDetector
from asl.data.preprocessor import ASLPreprocessor
from asl.models.coords_model import CoordsModel
from asl.visualization.visualizer import ASLVisualizer
import os

# Replace with path to your model
model_path = "path/to/model.pt"

if os.path.exists(model_path):
    # Create pipeline
    pipeline = ASLPipeline.load_pipeline(
        model_path,
        detector_params={"min_detection_confidence": 0.7},
        preprocessor_params={"normalize": True, "flatten": True}
    )
    
    # Run real-time inference
    pipeline.process_video(use_webcam=True)
else:
    print(f"Model not found at {model_path}")
