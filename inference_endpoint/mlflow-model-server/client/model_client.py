import requests
import numpy as np
import logging
import time

class ModelClient:
    """Simple client for MLflow Model Server"""
    
    def __init__(self, server_url="http://localhost:5001", timeout=30):
        """
        Initialize the client
        
        Args:
            server_url: URL of the model server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
    
    def health(self):
        """Check server health"""
        response = requests.get(
            f"{self.server_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def predict(self, model_name, data, version="latest"):
        """
        Run prediction with a model
        
        Args:
            model_name: Name of the model
            data: Input data for prediction (numpy array or list)
            version: Model version, or "latest"
            
        Returns:
            Prediction result
        """
        # Convert numpy arrays to lists
        if isinstance(data, np.ndarray):
            data = data.tolist()
        
        # Prepare request payload
        payload = {"data": data}
            
        # Make request
        start_time = time.time()
        response = requests.post(
            f"{self.server_url}/predict/{model_name}",
            json=payload,
            params={"version": version},
            timeout=self.timeout
        )
        
        # Handle response
        response.raise_for_status()
        return response.json()