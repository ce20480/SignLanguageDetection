import datetime
import json
import os

import numpy as np


class ContributionManager:
    """Handles storage and management of user contributions"""

    def __init__(self, storage_dir=None):
        """
        Initialize the contribution manager.

        Args:
            storage_dir (str, optional): Directory to store contributions
        """
        self.storage_dir = storage_dir or os.path.join(
            os.getcwd(), "data", "contributions"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

    def store_contribution(self, landmarks, label, metadata=None):
        """
        Store contributed landmark data for future model training.

        Args:
            landmarks (list or numpy.ndarray): Hand landmarks
            label (str): Sign language label
            metadata (dict, optional): Additional metadata

        Returns:
            bool: Success status
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{label}_{timestamp}.json"
            filepath = os.path.join(self.storage_dir, filename)

            # Ensure landmarks are in a serializable format
            if isinstance(landmarks, np.ndarray):
                landmarks = landmarks.tolist()

            # Prepare data
            data = {
                "landmarks": landmarks,
                "label": label,
                "timestamp": timestamp,
                "source": "user_contribution",
            }

            # Add any additional metadata
            if metadata:
                data.update(metadata)

            # Save to file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error storing contribution: {e}")
            return False
