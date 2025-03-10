import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    """Handles hand detection and landmark extraction"""
    
    def __init__(self, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the hand detector.
        
        Args:
            model_complexity (int): Model complexity (0, 1, or 2)
            min_detection_confidence (float): Minimum confidence for detection
            min_tracking_confidence (float): Minimum confidence for tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1
        )
    
    def detect_from_image(self, image):
        """
        Detect hands and extract landmarks from an image.
        
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
    
    def detect_from_webcam(self, display=True, preprocessor=None, model=None, visualizer=None):
        """
        Run hand detection on webcam feed.
        
        Args:
            display (bool): Whether to display the webcam feed
            preprocessor (object, optional): Preprocessor for landmarks
            model (object, optional): Model for prediction
            visualizer (object, optional): Visualizer for results
            
        Returns:
            None
        """
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, image = cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Process image
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # Draw landmarks and make predictions
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                # Extract landmarks
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                
                landmarks = np.array(landmarks)
                
                # Draw landmarks
                self.draw_landmarks(image, hand_landmarks)
                
                # Make prediction if model and preprocessor are provided
                if preprocessor and model:
                    # Preprocess landmarks
                    if preprocessor.normalize:
                        landmarks = preprocessor.normalize_landmarks(landmarks)
                    
                    if preprocessor.flatten:
                        features = preprocessor.flatten_landmarks(landmarks)
                    else:
                        features = landmarks
                    
                    # Make prediction
                    prediction = model.predict(features)
                    
                    # Visualize prediction if visualizer is provided
                    if visualizer:
                        image = visualizer.visualize_prediction(image, landmarks, prediction)
                    else:
                        # Simple text overlay
                        cv2.putText(
                            image,
                            f"Class: {prediction['class_index']} ({prediction['confidence']:.2f})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
            
            # Display the image
            if display:
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                
                if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                    break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_landmarks(self, image, landmarks):
        """
        Draw landmarks on an image.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Hand landmarks
            
        Returns:
            numpy.ndarray: Image with landmarks drawn
        """
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        return image
