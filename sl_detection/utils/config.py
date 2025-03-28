import os
import yaml
import json

class Config:
    """Configuration utility for ASL detection"""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration
        """
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif ext.lower() == '.json':
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        return self.config
    
    def save_config(self, config_path):
        """
        Save configuration to file.
        
        Args:
            config_path (str): Path to save configuration
        """
        _, ext = os.path.splitext(config_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        if ext.lower() in ['.yml', '.yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif ext.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
    
    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): Configuration key
            default: Default value if key is not found
            
        Returns:
            Value for the key
        """
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key (str): Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def update(self, config_dict):
        """
        Update configuration with a dictionary.
        
        Args:
            config_dict (dict): Dictionary to update with
        """
        self.config.update(config_dict)
