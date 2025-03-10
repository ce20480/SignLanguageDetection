import matplotlib.pyplot as plt
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ASLVisualizer:
    """Visualization tools for ASL detection"""
    
    def __init__(self, label_mapping=None):
        """
        Initialize the ASL visualizer.
        
        Args:
            label_mapping (dict, optional): Mapping from class indices to labels
        """
        self.label_mapping = label_mapping
    
    def plot_landmarks(self, landmarks, figsize=(10, 10)):
        """
        Plot hand landmarks in 3D.
        
        Args:
            landmarks (numpy.ndarray): Hand landmarks
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot landmarks
        ax.scatter(
            landmarks[:, 0],
            landmarks[:, 1],
            landmarks[:, 2],
            c=np.arange(landmarks.shape[0]),
            cmap='viridis',
            s=100
        )
        
        # Connect landmarks with lines
        # Thumb
        self._connect_landmarks(ax, landmarks, [0, 1, 2, 3, 4])
        # Index finger
        self._connect_landmarks(ax, landmarks, [0, 5, 6, 7, 8])
        # Middle finger
        self._connect_landmarks(ax, landmarks, [0, 9, 10, 11, 12])
        # Ring finger
        self._connect_landmarks(ax, landmarks, [0, 13, 14, 15, 16])
        # Pinky
        self._connect_landmarks(ax, landmarks, [0, 17, 18, 19, 20])
        # Palm
        self._connect_landmarks(ax, landmarks, [0, 5, 9, 13, 17])
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Hand Landmarks')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig
    
    def _connect_landmarks(self, ax, landmarks, indices):
        """
        Connect landmarks with lines.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to plot on
            landmarks (numpy.ndarray): Hand landmarks
            indices (list): Indices of landmarks to connect
        """
        for i in range(len(indices) - 1):
            ax.plot(
                [landmarks[indices[i], 0], landmarks[indices[i+1], 0]],
                [landmarks[indices[i], 1], landmarks[indices[i+1], 1]],
                [landmarks[indices[i], 2], landmarks[indices[i+1], 2]],
                'k-'
            )
    
    def visualize_prediction(self, image, landmarks, prediction):
        """
        Visualize prediction results on an image.
        
        Args:
            image (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Hand landmarks
            prediction (dict): Prediction results
            
        Returns:
            numpy.ndarray: Image with prediction visualization
        """
        # Create a copy of the image
        result_image = image.copy()
        
        # Get prediction information
        class_index = prediction['class_index']
        confidence = prediction['confidence']
        
        # Get class label if mapping is available
        if self.label_mapping and class_index in self.label_mapping:
            label = self.label_mapping[class_index]
        else:
            label = str(class_index)
        
        # Draw prediction text
        cv2.putText(
            result_image,
            f"Class: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            result_image,
            f"Confidence: {confidence:.2f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return result_image
    
    def plot_training_history(self, history):
        """
        Plot training metrics over epochs.
        
        Args:
            history (dict): Training history
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Training Accuracy')
        if 'val_acc' in history and history['val_acc']:
            ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        Plot confusion matrix of model predictions.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            class_names (list, optional): List of class names
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create display
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        return fig
