# Sign Language Detection

A real-time sign language detection system using computer vision and machine learning techniques.

## Overview

This repository contains a system for detecting and recognizing sign language gestures through a webcam. The system uses MediaPipe for hand landmark detection and a trained neural network model to classify these landmarks into sign language gestures.

## Key Components

### Core Files

- **CameraClass.py**: Main implementation of the `HandDetector` class that:

  - Captures real-time video from webcam
  - Detects hand landmarks using MediaPipe
  - Makes predictions using the trained model
  - Provides live visualization with bounding boxes

- **coordsnn.py**: PyTorch implementation of the neural network model (`CoordsNN`) that:

  - Takes hand landmark coordinates as input
  - Performs classification of sign gestures
  - Provides training and evaluation methods

- **get_coordinates.py**: Contains utility functions to extract hand landmarks from images using MediaPipe

- **handDetection.py**: Basic implementation of hand detection using OpenCV and MediaPipe

### Main Notebooks

- **ASLDetection.ipynb**: Complete pipeline for American Sign Language detection
- **main.ipynb**: Alternative implementation of the sign language detection system

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages include:

- OpenCV (cv2)
- MediaPipe
- TensorFlow/Keras
- PyTorch
- NumPy
- Pandas
- Matplotlib

### Running the Real-Time Sign Language Detector

1. The simplest way to use the system is through the `CameraClass.py`:

```python
from CameraClass import HandDetector

# Initialize the detector with path to your trained model
detector = HandDetector(model_path='path/to/your/model.h5')

# Start the live video detection
detector.live_video()
```

2. Alternatively, you can run one of the provided notebooks:
   - `ASLDetection.ipynb` for a comprehensive implementation
   - `main.ipynb` for an alternative approach

## Model Training

The system uses a neural network model trained on hand landmark coordinates. The training process involves:

1. Collecting sign language gesture data
2. Extracting hand landmarks using MediaPipe
3. Training the model with the landmark data

You can refer to the notebooks for examples of the training process.

## Notes

This repository contains additional experimental files that are not part of the core functionality. The files described above represent the key components for the sign language detection system.
