from .coords_model import CoordsModel

class ModelFactory:
    """Factory for creating different types of models"""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create a model of the specified type.
        
        Args:
            model_type (str): Type of model to create
            **kwargs: Model parameters
            
        Returns:
            object: Model instance
            
        Raises:
            ValueError: If model_type is unknown
        """
        if model_type.lower() == "coords":
            return CoordsModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
