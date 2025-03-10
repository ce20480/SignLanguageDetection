import torch
import numpy as np
import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asl.models.coords_model import CoordsModel
from asl.models.model_factory import ModelFactory

# Test model creation
input_size = 63  # 21 landmarks with 3 coordinates each
output_size = 27  # 26 letters + '0' class
hidden_layers = [(128, "RELU"), (64, "RELU")]

# Test model factory
model = ModelFactory.create_model(
    "coords",
    input_size=input_size,
    hidden_layers=hidden_layers,
    output_size=output_size
)

print("Model creation: PASSED")
print(f"Model architecture: {model.model}")

# Test forward pass with dummy data
dummy_input = torch.rand(10, input_size)  # Batch of 10 samples
output = model.model(dummy_input)

assert output.shape == (10, output_size), f"Expected output shape (10, {output_size}), got {output.shape}"
print("Forward pass: PASSED")

# Test save and load
model_dir = "data/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "test_model.pt")

model.save(model_path)
assert os.path.exists(model_path), f"Model not saved to {model_path}"
print(f"Model saved: PASSED")

loaded_model = CoordsModel.load(model_path)
assert loaded_model is not None, "Failed to load model"
print("Model loading: PASSED")

# Clean up test model
os.remove(model_path)
