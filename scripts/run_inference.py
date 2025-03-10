import os
import argparse
import cv2
import numpy as np

from asl.detection.hand_detector import HandDetector
from asl.data.preprocessor import ASLPreprocessor
from asl.models.coords_model import CoordsModel
from asl.visualization.visualizer import ASLVisualizer
from asl.pipeline import ASLPipeline

def main(args):
    """
    Run ASL detection inference.
    
    Args:
        args: Command-line arguments
    """
    print(f"Loading model from {args.model_path}...")
    
    # Create pipeline
    pipeline = ASLPipeline.load_pipeline(
        args.model_path,
        detector_params={"min_detection_confidence": 0.7},
        preprocessor_params={"normalize": True, "flatten": True}
    )
    
    if args.image_path:
        # Process single image
        print(f"Processing image {args.image_path}...")
        
        image = cv2.imread(args.image_path)
        
        if image is None:
            print(f"Error: Could not read image {args.image_path}")
            return
        
        prediction, landmarks, processed_image = pipeline.process_image(image)
        
        if prediction is None:
            print("No hand detected in the image")
            return
        
        print(f"Prediction: Class {prediction['class_index']} with confidence {prediction['confidence']:.4f}")
        
        # Display result
        cv2.imshow("ASL Detection", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result if output path is provided
        if args.output_path:
            cv2.imwrite(args.output_path, processed_image)
            print(f"Result saved to {args.output_path}")
    
    elif args.video_path:
        # Process video
        print(f"Processing video {args.video_path}...")
        
        pipeline.process_video(
            video_path=args.video_path,
            output_path=args.output_path
        )
    
    else:
        # Process webcam feed
        print("Processing webcam feed...")
        
        pipeline.process_video(
            use_webcam=True,
            output_path=args.output_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASL detection inference")
    
    # Input arguments
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--image_path", help="Path to image for inference")
    parser.add_argument("--video_path", help="Path to video for inference")
    
    # Output arguments
    parser.add_argument("--output_path", help="Path to save output")
    
    args = parser.parse_args()
    
    if not (args.image_path or args.video_path or args.webcam):
        args.webcam = True
    
    main(args)
