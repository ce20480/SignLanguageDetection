#!/usr/bin/env python3
"""
ASL Sign Language Detection - Training and Real-time Inference

This script:
1. Loads pre-processed ASL sign language data
2. Creates and trains a neural network model
3. Saves the trained model
4. Runs real-time inference using webcam input
"""

import os
import string
import sys
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import ASL modules
from asl.data.dataset import ASLDataset
from asl.data.preprocessor import ASLPreprocessor
from asl.detection.hand_detector import HandDetector
from asl.models.model_factory import ModelFactory
from asl.pipeline import ASLPipeline
from asl.visualization.visualizer import ASLVisualizer

# Configuration
DATA_DIR = "data/processed"  # Directory with processed data
MODEL_DIR = "data/models"  # Directory to save models
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for optimizer
HIDDEN_LAYERS = [(128, "RELU"), (64, "RELU")]  # Model architecture

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)


def create_label_mapping():
    """Create a mapping from class indices to letter labels"""
    mapping = {0: "0"}  # Class 0 is blank/neutral
    for i, letter in enumerate(string.ascii_uppercase):
        mapping[i + 1] = letter
    return mapping


def train_model(X_train, y_train, X_test, y_test):
    """Train a model on the provided data"""
    print("\n=== Training Model ===")

    # Get input and output dimensions
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Create model
    model = ModelFactory.create_model(
        "coords",
        input_size=input_size,
        hidden_layers=HIDDEN_LAYERS,
        output_size=output_size,
        learning_rate=LEARNING_RATE,
    )

    # Train model
    print(f"\nTraining for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    start_time = time.time()

    history = model.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"asl_model_{timestamp}.pt")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Visualize training history
    visualizer = ASLVisualizer()
    history_fig = visualizer.plot_training_history(history)

    # Save training history plot
    history_path = os.path.join(MODEL_DIR, f"training_history_{timestamp}.png")
    history_fig.savefig(history_path)
    print(f"Training history plot saved to {history_path}")

    plt.show()  # Display the plot

    return model, model_path


def run_webcam_inference(model_path, label_mapping):
    """Run real-time inference using webcam input"""
    print("\n=== Starting Real-time Inference ===")
    print("Press ESC to exit")

    # Create pipeline components
    detector = HandDetector(min_detection_confidence=0.7)
    preprocessor = ASLPreprocessor(normalize=True, flatten=True)

    # Load model
    from asl.models.coords_model import CoordsModel

    model = CoordsModel.load(model_path)

    # Create visualizer with label mapping
    visualizer = ASLVisualizer()

    # Create pipeline
    pipeline = ASLPipeline(detector, preprocessor, model, visualizer)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set up window
    cv2.namedWindow("ASL Sign Language Detection", cv2.WINDOW_NORMAL)

    # Create a mapping from class indices to letter labels
    letter_mapping = create_label_mapping()

    # Define colors for different finger landmarks
    colors = [
        (255, 0, 0),  # Thumb - Blue
        (0, 255, 0),  # Index - Green
        (0, 255, 255),  # Middle - Yellow
        (0, 165, 255),  # Ring - Orange
        (128, 0, 255),  # Pinky - Purple
    ]

    # Define connections between landmarks for drawing
    connections = [
        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        # Index finger
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        # Middle finger
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        # Ring finger
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        # Palm
        (0, 5),
        (5, 9),
        (9, 13),
        (13, 17),
    ]

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Process frame
        prediction, landmarks, processed_frame = pipeline.process_image(frame)

        # Add letter label to the frame if hand detected
        if prediction is not None:
            class_idx = prediction["class_index"]
            confidence = prediction["confidence"]
            letter = letter_mapping.get(class_idx, f"Unknown ({class_idx})")

            # Draw letter in a large, clear format
            cv2.putText(
                processed_frame,
                f"Letter: {letter}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                5,
            )

            cv2.putText(
                processed_frame,
                f"Confidence: {confidence:.2f}",
                (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Draw landmarks and coordinates if available
            if landmarks is not None:
                h, w, _ = processed_frame.shape

                # Draw landmarks with coordinates
                for i, (x, y, z) in enumerate(landmarks):
                    # Convert normalized coordinates to pixel coordinates
                    px, py = int(x * w), int(y * h)

                    # Determine color based on which finger the landmark belongs to
                    if i == 0:  # Wrist
                        color = (255, 255, 255)  # White
                    else:
                        # Determine which finger this landmark belongs to
                        finger_idx = (i - 1) // 4
                        if finger_idx < len(colors):
                            color = colors[finger_idx]
                        else:
                            color = (200, 200, 200)  # Gray fallback

                    # Draw circle at landmark position
                    cv2.circle(processed_frame, (px, py), 5, color, -1)

                    # Draw landmark index and coordinates
                    if (
                        i % 4 == 0
                    ):  # Only show coordinates for key points to avoid clutter
                        coord_text = f"{i}: ({x:.2f}, {y:.2f}, {z:.2f})"
                        cv2.putText(
                            processed_frame,
                            coord_text,
                            (px + 10, py),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                        )

                # Draw connections between landmarks
                for connection in connections:
                    start_idx, end_idx = connection
                    start_point = (
                        int(landmarks[start_idx][0] * w),
                        int(landmarks[start_idx][1] * h),
                    )
                    end_point = (
                        int(landmarks[end_idx][0] * w),
                        int(landmarks[end_idx][1] * h),
                    )

                    # Determine color based on which finger the connection belongs to
                    if start_idx == 0 and end_idx >= 5:  # Palm connections
                        color = (255, 255, 255)  # White
                    else:
                        # Get finger index
                        if end_idx <= 4:
                            finger_idx = 0  # Thumb
                        else:
                            finger_idx = (end_idx - 1) // 4

                        if finger_idx < len(colors):
                            color = colors[finger_idx]
                        else:
                            color = (200, 200, 200)  # Gray fallback

                    # Draw line
                    cv2.line(processed_frame, start_point, end_point, color, 2)
        else:
            # No hand detected
            cv2.putText(
                processed_frame,
                "No hand detected",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
            )

        # Display frame
        cv2.imshow("ASL Sign Language Detection", processed_frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function to run the training and inference pipeline"""
    print("=== ASL Sign Language Detection ===")

    # Load data
    print("\n=== Loading Data ===")
    dataset = ASLDataset()

    try:
        # Try to load from processed directory
        X_train, X_test, y_train, y_test = dataset.load_processed_data(DATA_DIR)
        print(f"Loaded pre-processed data from {DATA_DIR}")
    except Exception as e:
        print(f"Error loading pre-processed data: {e}")
        print(
            "Please make sure the processed data files exist in the specified directory."
        )
        return

    # Train or load model
    train_new_model = (
        input("\nDo you want to train a new model? (y/n): ").lower() == "y"
    )

    if train_new_model:
        # Train new model
        model, model_path = train_model(X_train, y_train, X_test, y_test)
    else:
        # Use existing model
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]

        if not model_files:
            print(f"No model files found in {MODEL_DIR}. Training a new model.")
            model, model_path = train_model(X_train, y_train, X_test, y_test)
        else:
            print("\nAvailable models:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")

            model_idx = int(input("\nSelect a model (number): ")) - 1
            model_path = os.path.join(MODEL_DIR, model_files[model_idx])
            print(f"Using model: {model_path}")

    # Create label mapping
    label_mapping = create_label_mapping()

    # Run webcam inference
    run_webcam_inference(model_path, label_mapping)


if __name__ == "__main__":
    main()
