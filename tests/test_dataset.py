import numpy as np
import string
import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asl.data.dataset import ASLDataset

# Test dataset initialization and label mapping
ASLdata = ASLDataset("data/raw")
x_train, x_test, y_train, y_test = ASLdata.load_processed_data("data/processed")

# Verify label mapping creation
# assert '0' in ASLdata.label_mapping, "Label mapping missing '0' class"
# for letter in string.ascii_uppercase:
#     assert letter in ASLdata.label_mapping, f"Label mapping missing '{letter}' class"

# print("Label mapping creation: PASSED")

# # Test one-hot encoding
# for letter in string.ascii_uppercase:
    # one_hot = ASLdata.label_mapping[letter]
#     assert len(one_hot) == 27, f"One-hot length incorrect for '{letter}'"
#     assert np.sum(one_hot) == 1, f"One-hot encoding invalid for '{letter}'"

# print("One-hot encoding: PASSED")
