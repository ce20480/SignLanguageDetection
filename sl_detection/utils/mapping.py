import string


def create_asl_letter_mapping():
    """
    Create a mapping from class indices to ASL letters.

    Returns:
        dict: Mapping from class indices to letters
    """
    # Map class indices to letters
    mapping = {0: "0"}  # Class 0 is blank/neutral
    for i, letter in enumerate(string.ascii_uppercase):
        mapping[i + 1] = letter
    return mapping


def get_letter_from_prediction(prediction, mapping=None):
    """
    Get the letter from a prediction using the mapping.

    Args:
        prediction (dict): Prediction with class_index
        mapping (dict, optional): Custom mapping to use

    Returns:
        str: The letter corresponding to the predicted class
    """
    if mapping is None:
        mapping = create_asl_letter_mapping()

    class_idx = prediction["class_index"]
    return mapping.get(class_idx, f"Unknown ({class_idx})")
