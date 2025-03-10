import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Example workflow with pre-processed data
from asl.data.dataset import ASLDataset
from asl.models.model_factory import ModelFactory
from asl.visualization.visualizer import ASLVisualizer

# 1. Load pre-processed data
dataset = ASLDataset()
X_train, X_test, y_train, y_test = dataset.load_processed_data("data/processed")

# 2. Create and train model
input_size = X_train.shape[1]  # Number of features
output_size = y_train.shape[1]  # Number of classes
hidden_layers = [(128, "RELU"), (64, "RELU")]

model = ModelFactory.create_model(
    "coords",
    input_size=input_size,
    hidden_layers=hidden_layers,
    output_size=output_size
)

# 3. Train model
history = model.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=100,
    batch_size=32
)

# 4. Visualize results
visualizer = ASLVisualizer()
visualizer.plot_training_history(history)

# 5. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 6. Save model
model.save("data/models/asl_model.pt")