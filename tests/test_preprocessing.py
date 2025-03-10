from asl.data.preprocessor import ASLPreprocessor
from asl.detection.hand_detector import HandDetector
import cv2
import numpy as np
import os

# Test with a sample image
# Replace with a path to a sample image that has a hand
sample_image_path = "path/to/sample_hand_image.jpg"

if os.path.exists(sample_image_path):
    # Load image
    image = cv2.imread(sample_image_path)
    assert image is not None, "Failed to load image"
    
    # Test detector
    detector = HandDetector()
    landmarks = detector.detect_from_image(image)
    
    if landmarks is not None:
        print("Hand detection: PASSED")
        print(f"Detected {len(landmarks)} landmarks")
        
        # Test preprocessor
        preprocessor = ASLPreprocessor(normalize=True, flatten=True)
        
        # Test landmark normalization
        normalized = preprocessor.normalize_landmarks(landmarks)
        assert normalized is not None, "Normalization failed"
        assert normalized.shape == landmarks.shape, "Normalization changed shape"
        
        # Check normalization actually worked
        assert np.min(normalized[:, 0]) >= 0 and np.max(normalized[:, 0]) <= 1, "X-normalization failed"
        assert np.min(normalized[:, 1]) >= 0 and np.max(normalized[:, 1]) <= 1, "Y-normalization failed"
        
        print("Landmark normalization: PASSED")
        
        # Test flattening
        flattened = preprocessor.flatten_landmarks(normalized)
        assert flattened is not None, "Flattening failed"
        assert len(flattened.shape) == 1, "Flattening didn't reduce dimensions"
        assert flattened.shape[0] == landmarks.shape[0] * landmarks.shape[1], "Flattened shape incorrect"
        
        print("Landmark flattening: PASSED")
        
        # Test full preprocessing pipeline
        processed = preprocessor.preprocess_image(image)
        assert processed is not None, "Full preprocessing failed"
        assert len(processed.shape) == 1, "Preprocessed output has wrong shape"
        
        print("Full preprocessing pipeline: PASSED")
    else:
        print("No hand detected in sample image. Please use an image with a visible hand.")
else:
    print(f"Sample image not found at {sample_image_path}")
