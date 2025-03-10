import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from asl.data.dataset import ASLDataset
from asl.data.preprocessor import ASLPreprocessor
from asl.models.model_factory import ModelFactory
from asl.visualization.visualizer import ASLVisualizer

def main(args):
    """
    Train an ASL detection model.
    
    Args:
        args: Command-line arguments
    """
    print(f"Loading dataset from {args.data_dir}...")
    
    # Create dataset
    dataset = ASLDataset(args.data_dir, test_size=args.test_size)
    
    # Load data
    if args.from_csv:
        X_train, X_test, y_train, y_test = dataset.load_from_csv(
            args.x_train_file, args.y_train_file
        )
    else:
        X_train, X_test, y_train, y_test = dataset.load_from_directory()
    
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
    
    # Create preprocessor
    preprocessor = ASLPreprocessor(normalize=True, flatten=True)
    
    print("Preprocessing data...")
    
    # Preprocess data
    X_train_proc = preprocessor.preprocess_batch(X_train)
    X_test_proc = preprocessor.preprocess_batch(X_test)
    
    print(f"Preprocessed data shape: {X_train_proc.shape}")
    
    # Create model
    hidden_layers = [(128, "RELU"), (64, "RELU")]
    
    print(f"Creating model with input size {X_train_proc.shape[1]} and output size {y_train.shape[1]}...")
    
    model = ModelFactory.create_model(
        "coords",
        input_size=X_train_proc.shape[1],
        hidden_layers=hidden_layers,
        output_size=y_train.shape[1],
        learning_rate=args.learning_rate
    )
    
    # Train model
    print(f"Training model for {args.epochs} epochs with batch size {args.batch_size}...")
    
    history = model.train(
        X_train_proc, y_train,
        X_val=X_test_proc, y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate model
    print("Evaluating model...")
    
    loss, accuracy = model.evaluate(X_test_proc, y_test)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model.pt")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Plot and save training history
        visualizer = ASLVisualizer()
        history_fig = visualizer.plot_training_history(history)
        history_path = os.path.join(args.output_dir, "training_history.png")
        history_fig.savefig(history_path)
        print(f"Training history plot saved to {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ASL detection model")
    
    # Data arguments
    parser.add_argument("--data_dir", required=True, help="Directory containing the dataset")
    parser.add_argument("--from_csv", action="store_true", help="Load data from CSV files")
    parser.add_argument("--x_train_file", help="Path to X train CSV file")
    parser.add_argument("--y_train_file", help="Path to y train CSV file")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    
    # Output arguments
    parser.add_argument("--output_dir", help="Directory to save model and results")
    
    args = parser.parse_args()
    main(args)
