import numpy as np
import cv2
import mediapipe as mp

class ASLPreprocessor:
    """Handles preprocessing of ASL data"""
    
    def __init__(self, normalize=True, flatten=True):
        """
        Initialize the ASL preprocessor.
        
        Args:
            normalize (bool): Whether to normalize landmark coordinates
            flatten (bool): Whether to flatten landmarks to 1D array
        """
        self.normalize = normalize
        self.flatten = flatten
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image):
        """
        Extract hand landmarks from an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list or None: List of landmarks if hand detected, None otherwise
        """
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmark coordinates.
        
        Args:
            landmarks (numpy.ndarray): Hand landmarks
            
        Returns:
            numpy.ndarray: Normalized landmarks
        """
        if landmarks is None:
            return None
            
        # Calculate bounding box
        min_x = np.min(landmarks[:, 0])
        max_x = np.max(landmarks[:, 0])
        min_y = np.min(landmarks[:, 1])
        max_y = np.max(landmarks[:, 1])
        
        # Calculate scale factors
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        if x_range == 0 or y_range == 0:
            return landmarks
        
        # Normalize coordinates to [0, 1] range
        normalized = landmarks.copy()
        normalized[:, 0] = (normalized[:, 0] - min_x) / x_range
        normalized[:, 1] = (normalized[:, 1] - min_y) / y_range
        
        return normalized
    
    def flatten_landmarks(self, landmarks):
        """
        Flatten 3D landmarks to 1D feature vector.
        
        Args:
            landmarks (numpy.ndarray): Hand landmarks
            
        Returns:
            numpy.ndarray: Flattened landmarks
        """
        if landmarks is None:
            return None
            
        return landmarks.flatten()
    
    def preprocess_image(self, image):
        """
        Preprocess a single image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray or None: Preprocessed landmarks
        """
        # Extract landmarks
        landmarks = self.extract_landmarks(image)
        
        if landmarks is None:
            return None
        
        # Normalize if required
        if self.normalize:
            landmarks = self.normalize_landmarks(landmarks)
        
        # Flatten if required
        if self.flatten:
            landmarks = self.flatten_landmarks(landmarks)
        
        return landmarks
    
    def preprocess_batch(self, images):
        """
        Preprocess a batch of images.
        
        Args:
            images (list or numpy.ndarray): Batch of images
            
        Returns:
            numpy.ndarray: Batch of preprocessed landmarks
        """
        processed = []
        
        for image in images:
            landmarks = self.preprocess_image(image)
            
            if landmarks is not None:
                processed.append(landmarks)
            else:
                # If no landmarks detected, use zeros
                if self.flatten:
                    # 21 landmarks with x, y, z coordinates
                    processed.append(np.zeros(21 * 3))
                else:
                    processed.append(np.zeros((21, 3)))
        
        return np.array(processed)
