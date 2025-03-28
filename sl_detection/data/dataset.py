import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import string

class ASLDataset:
    """Handles ASL dataset loading, splitting, and transformations"""
    
    def __init__(self, data_dir=None, test_size=0.2, random_state=42):
        """
        Initialize the ASL dataset handler.
        
        Args:
            data_dir (str, optional): Directory containing the data
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Initialize one-hot encoder
        categories = [[l] for l in list(string.ascii_uppercase)]
        categories.insert(0, ['0'])  # Add '0' class
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(categories)
        
        # Create label mapping
        self.label_mapping = {}
        onehot = self.encoder.transform(categories)
        self.label_mapping['0'] = onehot[0]
        for i, k in enumerate(string.ascii_uppercase):
            self.label_mapping[k] = onehot[i+1]
    
    def _listdir_nohidden(self, path):
        """List directory contents excluding hidden files"""
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f
    
    def load_from_directory(self, transform=None):
        """
        Load data from directory structure with class folders.
        
        Args:
            transform (callable, optional): Optional transform to apply to images
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) - Train/test split data
        """
        if self.data_dir is None:
            raise ValueError("data_dir must be specified to load from directory")
            
        images = []
        labels = []
        
        # Iterate through class folders
        for class_folder in self._listdir_nohidden(self.data_dir):
            class_path = os.path.join(self.data_dir, class_folder)
            
            if not os.path.isdir(class_path):
                continue
                
            # Iterate through images in the class folder
            for file in self._listdir_nohidden(class_path):
                image_path = os.path.join(class_path, file)
                
                # Read and preprocess image
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                # Apply transform if provided
                if transform:
                    image = transform(image)
                
                images.append(image)
                labels.append(self.label_mapping[class_folder])
        
        # Split data into training and testing sets
        if len(images) > 0:
            return self.split_data(np.array(images), np.array(labels))
        else:
            raise ValueError("No images found in the data directory")
    
    def load_from_csv(self, x_train_file, y_train_file, x_test_file=None, y_test_file=None):
        """
        Load preprocessed data from CSV files.
        
        Args:
            x_train_file (str): Path to training features CSV file
            y_train_file (str): Path to training labels CSV file
            x_test_file (str, optional): Path to test features CSV file
            y_test_file (str, optional): Path to test labels CSV file
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Load training data
        self.X_train = pd.read_csv(x_train_file).values
        self.y_train = pd.read_csv(y_train_file).values
        
        # Load test data if provided
        if x_test_file and y_test_file:
            self.X_test = pd.read_csv(x_test_file).values
            self.y_test = pd.read_csv(y_test_file).values
        else:
            # Split training data if test files not provided
            return self.split_data(self.X_train, self.y_train)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def load_processed_data(self, processed_dir):
        """
        Load preprocessed data from a directory containing train/test split files.
        
        Args:
            processed_dir (str): Directory containing processed data files
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Define file paths
        x_train_path = os.path.join(processed_dir, "x_train.csv")
        y_train_path = os.path.join(processed_dir, "y_train.csv")
        x_test_path = os.path.join(processed_dir, "x_test.csv")
        y_test_path = os.path.join(processed_dir, "y_test.csv")
        
        # Check if files exist
        if not all(os.path.exists(f) for f in [x_train_path, y_train_path, x_test_path, y_test_path]):
            raise FileNotFoundError(f"One or more data files not found in {processed_dir}")
        
        # Load data
        return self.load_from_csv(x_train_path, y_train_path, x_test_path, y_test_path)
    
    def split_data(self, X, y):
        """
        Split data into training and testing sets.
        
        Args:
            X (numpy.ndarray): Features
            y (numpy.ndarray): Labels
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_data(self, output_dir, prefix=""):
        """
        Save processed data to CSV files.
        
        Args:
            output_dir (str): Directory to save files
            prefix (str, optional): Prefix for filenames
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("No data to save. Load or split data first.")
        
        # Save training data
        pd.DataFrame(self.X_train).to_csv(
            os.path.join(output_dir, f"{prefix}x_train.csv"), index=False
        )
        pd.DataFrame(self.y_train).to_csv(
            os.path.join(output_dir, f"{prefix}y_train.csv"), index=False
        )
        
        # Save testing data
        pd.DataFrame(self.X_test).to_csv(
            os.path.join(output_dir, f"{prefix}x_test.csv"), index=False
        )
        pd.DataFrame(self.y_test).to_csv(
            os.path.join(output_dir, f"{prefix}y_test.csv"), index=False
        )
        
        print(f"Data saved to {output_dir}")
