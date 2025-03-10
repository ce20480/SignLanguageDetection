
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordsNN(nn.Module):
    def __init__(self, input_size, config, output_size, learning_rate=0.001):
        super(CoordsNN, self).__init__()
        self.input_size = input_size
        self.config = config
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()  # Use BCEWithLogitsLoss
        self.model = self.build_model()

    def forward(self, x):
        return self.model(x)

    def build_model(self):
        # Initialize model
        model = nn.Sequential()
        model.add_module('input', nn.Linear(self.input_size, self.config[0][0]))

        for i in range(1, len(self.config)):
            model.add_module(f'hidden_{i-1}', nn.Linear(self.config[i-1][0], self.config[i][0]))
            if self.config[i][-1] == 'RELU':
                model.add_module(f'activation_{i-1}', nn.ReLU())

        # Output layer
        model.add_module('output', nn.Linear(self.config[-1][0], self.output_size))
        
        return model

    def fit(self, x, y, epochs=100, batch_size=32, optimizer='Adam', learning_rate=0.001, val_data=None):
        training_losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(epochs):
            train_loss = 0
            correct_predictions = 0  # Initialize correct predictions counter
            total_samples = 0  # Initialize total samples counter
            num_batches = 0

            for i in range(0, len(x), batch_size):
                x_batch = x[i: i + batch_size]
                y_batch = y[i: i + batch_size]

                self.model.train()

                # Forward pass
                y_pred = self.model(x_batch)

                # Calculate the loss using one-hot encoded labels
                loss = self.loss_function(y_pred, y_batch)

                # Zero the gradients
                optimizer.zero_grad()

                # Backpropagation
                loss.backward()

                # Optimizer step
                optimizer.step()

                num_batches += 1
                train_loss += loss.item()

                # Calculate accuracy for the batch
                probabilities = F.softmax(y_pred, dim=1)  # Apply softmax to get probabilities
                predicted_classes = torch.argmax(probabilities, dim=1)  # Get predicted class indices
                true_classes = torch.argmax(y_batch, dim=1)  # Convert one-hot to class indices

                # Count correct predictions
                correct_predictions += (predicted_classes == true_classes).sum().item()  # Update correct count
                total_samples += y_batch.size(0)  # Update total samples count

            avg_loss = train_loss / num_batches  # Average loss for the epoch
            avg_accuracy = correct_predictions / total_samples  # Calculate average accuracy for the epoch

            training_losses.append(avg_loss)

            print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f}')

            if val_data:
                val_loss, val_acc = self.evaluate(val_data[0], val_data[1])
                print(f'Epoch {epoch + 1} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

    def predict(self, x):
        # single prediction
        self.model.eval()
        x = x.unsqueeze(0)
        with torch.no_grad():
            y_pred = self.model(x)
        probabilities = F.softmax(y_pred, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes


    def evaluate(self, x, y):
        # Set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Get logits from the model
            y_pred = self.model(x)
            
            # Calculate the loss using one-hot encoded labels
            loss = self.loss_function(y_pred, y)
            
            # Apply Softmax to convert logits to probabilities
            probabilities = F.softmax(y_pred, dim=1)  # Apply softmax along the class dimension
            
            # Get the predicted class indices (the class with the highest probability)
            predicted_classes = torch.argmax(probabilities, dim=1)  # Get class labels
            
            # Convert one-hot encoded y to class indices for accuracy calculation
            true_classes = torch.argmax(y, dim=1)  # Convert y from one-hot to class indices
            
            # Count correct predictions
            correct = (predicted_classes == true_classes).sum().item()  # Count correct predictions
            accuracy = correct / y.size(0)  # Calculate accuracy

            # Uncomment this line to see predicted probabilities and true labels
            # print("Predicted probabilities:", probabilities)
            # print("True labels:", y)

        return loss.item(), accuracy

