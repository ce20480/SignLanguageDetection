# ASL Sign Language Detection

A modular framework for American Sign Language (ASL) detection using computer vision and deep learning. This project enables real-time detection and classification of ASL hand signs using a webcam.

## Project Overview

This project implements a complete pipeline for detecting and recognizing American Sign Language (ASL) hand signs in real-time. The system uses hand landmark detection to extract features from hand poses and a neural network to classify these features into corresponding ASL letters.

### Key Features

- **Modular Architecture**: Clean separation of concerns with specialized components
- **End-to-End Pipeline**: From raw images to real-time sign detection
- **Real-time Inference**: Webcam-based detection with visual feedback
- **Extensible Design**: Easy to add new signs or modify existing components

## How It Works

### 1. Data Collection and Preprocessing

The system uses MediaPipe's hand landmark detection to extract 21 3D landmarks from hand images. Each landmark has (x, y, z) coordinates, resulting in 63 features per hand pose.

**Data Processing Steps:**

1. **Image Collection**: Images are organized by ASL letter class (A-Z)
2. **Hand Detection**: MediaPipe identifies hand regions in images
3. **Landmark Extraction**: 21 key points are extracted from each hand
4. **Normalization**: Coordinates are scaled to [0,1] range for consistency
5. **Flattening**: 3D landmarks are converted to a 1D feature vector
6. **Train/Test Split**: Data is divided for training and evaluation

**Code Example - Preprocessing:**

```python
# Extract and preprocess landmarks from an image
preprocessor = ASLPreprocessor(normalize=True, flatten=True)
landmarks = detector.detect_from_image(image)
features = preprocessor.preprocess_image(image)
```

### 2. Model Architecture

The model uses a fully connected neural network to classify hand landmark coordinates:

- **Input Layer**: 63 neurons (21 landmarks × 3 coordinates)
- **Hidden Layers**: Configurable fully connected layers with ReLU activation
- **Output Layer**: 27 neurons (26 ASL letters + neutral class)

The default architecture uses two hidden layers (128 and 64 neurons) with ReLU activation.

**Code Example - Model Creation:**

```python
# Create model with specified architecture
model = ModelFactory.create_model(
    "coords",
    input_size=63,  # 21 landmarks × 3 coordinates
    hidden_layers=[(128, "RELU"), (64, "RELU")],
    output_size=27  # 26 letters + neutral class
)
```

### 3. Training Process

The training process optimizes the model to correctly classify hand poses:

1. **Data Loading**: Preprocessed landmark data is loaded
2. **Batch Processing**: Data is processed in batches for efficiency
3. **Forward Pass**: Input features are passed through the network
4. **Loss Calculation**: Cross-entropy loss measures prediction error
5. **Backpropagation**: Gradients are calculated and propagated backward
6. **Parameter Updates**: Model weights are adjusted using Adam optimizer
7. **Validation**: Performance is monitored on a separate validation set
8. **Model Saving**: The trained model is saved for later use

**Code Example - Training:**

```python
# Train model with training and validation data
history = model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=50,
    batch_size=32
)
```

### 4. Inference Pipeline

The inference pipeline processes webcam frames in real-time:

1. **Frame Capture**: Webcam frames are captured using OpenCV
2. **Hand Detection**: MediaPipe detects hands in the frame
3. **Landmark Extraction**: 21 landmarks are extracted from detected hands
4. **Preprocessing**: Landmarks are normalized and flattened
5. **Classification**: The model predicts the ASL letter from landmarks
6. **Visualization**: Prediction results and landmarks are displayed

**Code Example - Inference:**

```python
# Process a single image through the pipeline
prediction, landmarks, processed_image = pipeline.process_image(image)
```

## Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/SignLanguageDetection.git
cd SignLanguageDetection
```

2. Install the package in development mode:

```bash
pip install -e .
```

3. Install required dependencies:

```bash
pip install numpy opencv-python torch matplotlib scikit-learn mediapipe
```

### Directory Structure

```
SignLanguageDetection/
├── asl/                           # Main package
│   ├── data/                      # Data management
│   │   ├── dataset.py             # Dataset creation and loading
│   │   └── preprocessor.py        # Data preprocessing utilities
│   ├── models/                    # Model definitions
│   │   ├── coords_model.py        # Coordinates-based neural network
│   │   └── model_factory.py       # Factory for creating different models
│   ├── detection/                 # Hand detection components
│   │   └── hand_detector.py       # Hand detection and landmark extraction
│   ├── visualization/             # Visualization utilities
│   │   └── visualizer.py          # Tools for visualizing results
│   ├── utils/                     # Utility functions
│   │   └── config.py              # Configuration settings
│   └── pipeline.py                # End-to-end pipeline
├── scripts/                       # Executable scripts
│   ├── train_model.py             # Script for training models
│   ├── run_inference.py           # Script for running inference
│   └── train_and_infer.py         # Combined training and inference script
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── inference_demo.ipynb
├── data/                          # Data directory
│   ├── raw/                       # Raw data
│   ├── processed/                 # Processed data
│   └── models/                    # Saved models
├── tests/                         # Unit tests
├── setup.py                       # Package installation
└── README.md                      # Project documentation
```

### Using Your Own Data

To use your own ASL sign language data:

1. **Organize Raw Data**: Place your images in the `data/raw` directory, organized by class:

```
data/raw/
├── A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── B/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

2. **Preprocess Data**: Use the data exploration notebook or preprocessing script:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

3. **Train Model**: Train a model on your preprocessed data:

```bash
python scripts/train_model.py --data_dir data/processed --output_dir data/models
```

4. **Run Inference**: Use your trained model for real-time inference:

```bash
python scripts/run_inference.py --model_path data/models/your_model.pt
```

### All-in-One Script

For convenience, you can use the all-in-one script to train and run inference:

```bash
python scripts/train_and_infer.py
```

This script will:

1. Load preprocessed data
2. Ask if you want to train a new model or use an existing one
3. If training, train and save a new model
4. Launch webcam inference with the selected model

## Component Details

### ASLDataset

Handles dataset loading, splitting, and transformations:

```python
# Load data from a processed directory
dataset = ASLDataset()
X_train, X_test, y_train, y_test = dataset.load_processed_data("data/processed")
```

### ASLPreprocessor

Extracts and preprocesses hand landmarks from images:

```python
# Create preprocessor
preprocessor = ASLPreprocessor(normalize=True, flatten=True)

# Preprocess a batch of images
features = preprocessor.preprocess_batch(images)
```

### HandDetector

Detects hands and extracts landmarks using MediaPipe:

```python
# Create detector
detector = HandDetector(min_detection_confidence=0.7)

# Detect hands in an image
landmarks = detector.detect_from_image(image)
```

### CoordsModel

Neural network model for hand coordinate classification:

```python
# Create model
model = CoordsModel(
    input_size=63,
    hidden_layers=[(128, "RELU"), (64, "RELU")],
    output_size=27
)

# Train model
history = model.train(X_train, y_train, epochs=50)

# Make prediction
prediction = model.predict(features)
```

### ASLVisualizer

Tools for visualizing hand landmarks, predictions, and training metrics:

```python
# Create visualizer
visualizer = ASLVisualizer()

# Plot landmarks
fig = visualizer.plot_landmarks(landmarks)

# Visualize prediction
result_image = visualizer.visualize_prediction(image, landmarks, prediction)
```

### ASLPipeline

End-to-end pipeline for ASL detection:

```python
# Create pipeline
pipeline = ASLPipeline(detector, preprocessor, model, visualizer)

# Process image
prediction, landmarks, processed_image = pipeline.process_image(image)

# Run webcam inference
pipeline.process_video(use_webcam=True)
```

## Customization

### Model Architecture

You can customize the model architecture by modifying the hidden layers:

```python
# Example: Deeper network with different layer sizes
hidden_layers = [
    (256, "RELU"),
    (128, "RELU"),
    (64, "RELU")
]

model = ModelFactory.create_model(
    "coords",
    input_size=63,
    hidden_layers=hidden_layers,
    output_size=27
)
```

### Training Parameters

Adjust training parameters to improve model performance:

```python
# Example: Fine-tuning training parameters
history = model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=100,
    batch_size=64,
    learning_rate=0.0005
)
```

### Detection Sensitivity

Adjust hand detection sensitivity for different environments:

```python
# Example: Adjusting detection parameters
detector = HandDetector(
    model_complexity=1,  # 0, 1, or 2 (higher is more accurate but slower)
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
```

## Extending the Project

### Adding New Signs

To add support for new signs:

1. Create a new folder in `data/raw` for each new sign
2. Add images of the new signs to their respective folders
3. Retrain the model with the expanded dataset

### Implementing Sequence Recognition

To recognize sequences of signs (words or phrases):

1. Extend the pipeline to track signs over time
2. Implement a temporal model (e.g., LSTM or GRU)
3. Add language modeling for improved prediction

### Deploying to Mobile Devices

To deploy on mobile devices:

1. Export the model to a mobile-friendly format (e.g., TorchScript, ONNX)
2. Optimize the model for mobile performance
3. Integrate with mobile camera APIs

## Troubleshooting

### Common Issues

1. **Import Errors**: If you encounter import errors, make sure the package is installed in development mode:

```bash
pip install -e .
```

2. **Hand Detection Issues**: If hand detection is unreliable:

   - Ensure good lighting conditions
   - Position your hand clearly in the frame
   - Adjust the `min_detection_confidence` parameter

3. **Low Accuracy**: If the model has low accuracy:
   - Collect more training data
   - Try different model architectures
   - Adjust training parameters (epochs, learning rate)

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. Add support for more ASL signs
2. Improve model accuracy
3. Optimize for mobile devices
4. Add sequence recognition for words and phrases
5. Create a user-friendly GUI

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand landmark detection
- [PyTorch](https://pytorch.org/) for neural network implementation
- [OpenCV](https://opencv.org/) for image processing and visualization

You can refer to the notebooks for examples of the training process.

## Notes

This repository contains additional experimental files that are not part of the core functionality. The files described above represent the key components for the sign language detection system.
