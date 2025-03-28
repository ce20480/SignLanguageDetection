import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordsModel:
    """Neural network model for hand coordinate classification"""
    
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.001):
        """
        Initialize the coordinates model.
        
        Args:
            input_size (int): Size of input features
            hidden_layers (list): List of tuples (size, activation) for hidden layers
            output_size (int): Number of output classes
            learning_rate (float): Learning rate for optimizer
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        
        # Initialize loss function and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _build_model(self):
        """
        Build the neural network architecture.
        
        Returns:
            torch.nn.Sequential: The model
        """
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_size, self.hidden_layers[0][0]))
        
        # Add activation for input layer
        if self.hidden_layers[0][1] == "RELU":
            layers.append(nn.ReLU())
        elif self.hidden_layers[0][1] == "SIGMOID":
            layers.append(nn.Sigmoid())
        elif self.hidden_layers[0][1] == "TANH":
            layers.append(nn.Tanh())
        
        # Hidden layers
        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i-1][0], self.hidden_layers[i][0]))
            
            # Add activation
            if self.hidden_layers[i][1] == "RELU":
                layers.append(nn.ReLU())
            elif self.hidden_layers[i][1] == "SIGMOID":
                layers.append(nn.Sigmoid())
            elif self.hidden_layers[i][1] == "TANH":
                layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(self.hidden_layers[-1][0], self.output_size))
        
        return nn.Sequential(*layers)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the model on the given data.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray, optional): Validation features
            y_val (numpy.ndarray, optional): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            has_validation = True
        else:
            has_validation = False
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct_predictions = 0
            total_samples = 0
            
            # Process in batches
            for i in range(0, len(X_train), batch_size):
                # Get batch
                X_batch = X_train_tensor[i:i+batch_size]
                y_batch = y_train_tensor[i:i+batch_size]
                
                # Forward pass
                outputs = self.model(X_batch)
                
                # Calculate loss
                loss = self.loss_function(outputs, torch.argmax(y_batch, dim=1))
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
                total_samples += y_batch.size(0)
            
            # Calculate average loss and accuracy
            avg_train_loss = train_loss / (len(X_train) / batch_size)
            train_accuracy = correct_predictions / total_samples
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_accuracy)
            
            # Validation
            if has_validation:
                val_loss, val_accuracy = self.evaluate(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_accuracy)
                
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Acc: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Acc: {train_accuracy:.4f}")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            tuple: (loss, accuracy)
        """
        # Convert numpy arrays to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(X_test_tensor)
            
            # Calculate loss
            loss = self.loss_function(outputs, torch.argmax(y_test_tensor, dim=1))
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == torch.argmax(y_test_tensor, dim=1)).sum().item()
            accuracy = correct / len(y_test)
        
        return loss.item(), accuracy
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            dict: Prediction results with label and confidence
        """
        # Convert input to tensor
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(X)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
        
        return {
            'class_index': predicted_class.item(),
            'confidence': confidence.item()
        }
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'history': self.history
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            CoordsModel: Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(filepath)
        
        # Create model instance
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_layers=checkpoint['hidden_layers'],
            output_size=checkpoint['output_size'],
            learning_rate=checkpoint['learning_rate']
        )
        
        # Load model state
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.history = checkpoint['history']
        
        return model
