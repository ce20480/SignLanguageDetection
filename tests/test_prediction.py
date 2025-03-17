import torch
import pandas as pd
import numpy as np

def predict_from_file(model_path, input_file, device='cpu'):
    """
    Load a trained model and make predictions on input data from a file.
    
    Args:
        model_path (str): Path to the saved model (.pt file)
        input_file (str): Path to input data file (.csv or .npy)
        device (str): Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        tuple: (predicted_class_index, confidence_score)
    """
    # Load model checkpoint with weights_only=True
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Create model instance
    model = torch.nn.Sequential(
        *[torch.nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'][0][0]),
          torch.nn.ReLU(),
          torch.nn.Linear(checkpoint['hidden_layers'][0][0], checkpoint['hidden_layers'][1][0]),
          torch.nn.ReLU(),
          torch.nn.Linear(checkpoint['hidden_layers'][1][0], checkpoint['output_size'])]
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess input data
    if input_file.endswith('.csv'):
        # Read only the feature columns (assuming labels are in the last column)
        df = pd.read_csv(input_file)
        # Remove any non-numeric columns (like labels) if present
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        input_data = df[numeric_cols].values
    elif input_file.endswith('.npy'):
        input_data = np.load(input_file)
    else:
        raise ValueError("Input file must be .csv or .npy")
    
    # Ensure input data is float32
    input_data = input_data.astype(np.float32)
    
    # If we have multiple samples, we can either:
    # 1. Process one sample at a time:
    predictions = []
    confidences = []
    
    for sample in input_data:
        # Add batch dimension and convert to tensor
        input_tensor = torch.tensor(sample).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1).values
            
            predictions.append(pred_class.item())
            confidences.append(confidence.item())
    
    return predictions, confidences

def test_model_prediction():
    """Test model prediction on a sample input file"""
    model_path = "data/models/asl_model_20250310_184213.pt"
    input_file = "data/processed/x_test.csv"
    
    try:
        predictions, confidences = predict_from_file(model_path, input_file)
        
        print("\nPredictions for all samples:")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            print(f"Sample {i+1}:")
            print(f"  Predicted class index: {pred}")
            print(f"  Confidence: {conf:.4f}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    test_model_prediction()