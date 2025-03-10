import os
import numpy as np
import cv2
import torch

class ASLPipeline:
    """End-to-end pipeline for ASL detection"""
    
    def __init__(self, detector, preprocessor, model, visualizer=None):
        """
        Initialize the ASL pipeline.
        
        Args:
            detector (HandDetector): Hand detector
            preprocessor (ASLPreprocessor): Data preprocessor
            model (CoordsModel): Classification model
            visualizer (ASLVisualizer, optional): Visualizer
        """
        self.detector = detector
        self.preprocessor = preprocessor
        self.model = model
        self.visualizer = visualizer
    
    def process_image(self, image):
        """
        Process a single image through the pipeline.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (prediction, landmarks, processed_image)
        """
        # Detect hand landmarks
        landmarks = self.detector.detect_from_image(image)
        
        if landmarks is None:
            return None, None, image
        
        # Preprocess landmarks
        if self.preprocessor.normalize:
            normalized_landmarks = self.preprocessor.normalize_landmarks(landmarks)
        else:
            normalized_landmarks = landmarks
        
        if self.preprocessor.flatten:
            features = self.preprocessor.flatten_landmarks(normalized_landmarks)
        else:
            features = normalized_landmarks
        
        # Make prediction
        prediction = self.model.predict(features)
        
        # Visualize results
        if self.visualizer:
            processed_image = self.visualizer.visualize_prediction(image, landmarks, prediction)
        else:
            processed_image = image
        
        return prediction, landmarks, processed_image
    
    def process_video(self, video_path=None, use_webcam=False, output_path=None):
        """
        Process video input through the pipeline.
        
        Args:
            video_path (str, optional): Path to video file
            use_webcam (bool): Whether to use webcam input
            output_path (str, optional): Path to save output video
            
        Returns:
            None
        """
        # Set up video capture
        if use_webcam:
            cap = cv2.VideoCapture(0)
        elif video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            raise ValueError("Either video_path or use_webcam must be specified")
        
        # Set up video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                break
            
            # Process frame
            prediction, landmarks, processed_frame = self.process_image(frame)
            
            # Display frame
            cv2.imshow('ASL Detection', processed_frame)
            
            # Write frame if output path is provided
            if output_path:
                writer.write(processed_frame)
            
            # Exit on ESC key
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        # Clean up
        cap.release()
        if output_path:
            writer.release()
        cv2.destroyAllWindows()
    
    def train_pipeline(self, dataset_path, save_model_path=None, epochs=100, batch_size=32):
        """
        Train the pipeline on a dataset.
        
        Args:
            dataset_path (str): Path to dataset
            save_model_path (str, optional): Path to save trained model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        from asl.data.dataset import ASLDataset
        
        # Load dataset
        dataset = ASLDataset(dataset_path)
        X_train, X_test, y_train, y_test = dataset.load_from_directory()
        
        # Preprocess data
        X_train_proc = np.array([
            self.preprocessor.preprocess_image(img) for img in X_train
        ])
        X_test_proc = np.array([
            self.preprocessor.preprocess_image(img) for img in X_test
        ])
        
        # Train model
        history = self.model.train(
            X_train_proc, y_train,
            X_val=X_test_proc, y_val=y_test,
            epochs=epochs, batch_size=batch_size
        )
        
        # Save model if path is provided
        if save_model_path:
            self.model.save(save_model_path)
        
        return history
    
    def evaluate_pipeline(self, test_data_path):
        """
        Evaluate the pipeline on test data.
        
        Args:
            test_data_path (str): Path to test data
            
        Returns:
            tuple: (loss, accuracy)
        """
        from asl.data.dataset import ASLDataset
        
        # Load test data
        dataset = ASLDataset(test_data_path)
        _, X_test, _, y_test = dataset.load_from_directory()
        
        # Preprocess data
        X_test_proc = np.array([
            self.preprocessor.preprocess_image(img) for img in X_test
        ])
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test_proc, y_test)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    @classmethod
    def load_pipeline(cls, model_path, detector_params=None, preprocessor_params=None):
        """
        Load a pipeline with a pre-trained model.
        
        Args:
            model_path (str): Path to saved model
            detector_params (dict, optional): Parameters for hand detector
            preprocessor_params (dict, optional): Parameters for preprocessor
            
        Returns:
            ASLPipeline: Loaded pipeline
        """
        from asl.models.coords_model import CoordsModel
        from asl.detection.hand_detector import HandDetector
        from asl.data.preprocessor import ASLPreprocessor
        from asl.visualization.visualizer import ASLVisualizer
        
        # Load model
        model = CoordsModel.load(model_path)
        
        # Create detector
        if detector_params is None:
            detector_params = {}
        detector = HandDetector(**detector_params)
        
        # Create preprocessor
        if preprocessor_params is None:
            preprocessor_params = {}
        preprocessor = ASLPreprocessor(**preprocessor_params)
        
        # Create visualizer
        visualizer = ASLVisualizer()
        
        # Create and return pipeline
        return cls(detector, preprocessor, model, visualizer)
