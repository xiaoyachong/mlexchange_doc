import logging
import os

class ModelClient:
    def __init__(self, server_url="http://localhost:5001", timeout=30):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        
        # Import requests inside the class to avoid import issues
        import requests
        self.requests = requests
    
    def predict(self, model_name, data, version="latest"):
        """Run prediction with a model"""
        # Convert numpy arrays to lists
        if hasattr(data, 'tolist'):
            data = data.tolist()
        
        # Prepare request payload
        payload = {"data": data}
            
        # Make request
        response = self.requests.post(
            f"{self.server_url}/predict/{model_name}",
            json=payload,
            params={"version": version},
            timeout=self.timeout
        )
        
        # Handle response
        response.raise_for_status()
        return response.json()
    
    def health(self):
        """Check server health"""
        response = self.requests.get(
            f"{self.server_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

class MLflowClient:
    """A wrapper class for MLflow client operations."""
    
    def __init__(
        self, 
        model_server_url=None,
        tracking_uri=None,
        username=None, 
        password=None,
        cache_dir=None
    ):
        """Initialize the MLflow client with connection parameters."""
        self.model_server_url = model_server_url or os.getenv("MODEL_SERVER_URL", "http://model-server:5001")
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.username = username or os.getenv("MLFLOW_TRACKING_USERNAME", "")
        self.password = password or os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Try to connect to model server
        self.model_client = ModelClient(self.model_server_url)
        try:
            self.model_client.health()
            self.server_available = True
            self.logger.info(f"Connected to model server at {self.model_server_url}")
        except Exception as e:
            self.server_available = False
            self.logger.warning(f"Model server not available: {e}, falling back to direct MLflow")
            
            # Initialize MLflow client as fallback
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set environment variables
            os.environ['MLFLOW_TRACKING_USERNAME'] = self.username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = self.password
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create client
            self.client = MlflowClient()

    def check_mlflow_ready(self):
        """Check if MLflow is reachable."""
        if self.server_available:
            try:
                self.model_client.health()
                return True
            except Exception:
                self.server_available = False
                self.logger.warning("Model server unavailable, falling back to direct MLflow")
        
        try:
            # Try to use the MLflow client
            import mlflow
            return True
        except Exception as e:
            self.logger.warning(f"MLflow is not reachable: {e}")
            return False

    def load_model(self, model_name):
        """
        Load a model from MLflow by name with disk caching
        
        Args:
            model_name: Name of the model in MLflow
            
        Returns:
            The loaded model or None if loading fails
        """
        if model_name is None:
            self.logger.error("Cannot load model: model_name is None")
            return None
        
        if self.server_available:
            # Create a model wrapper that uses the server
            class ModelWrapper:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name
                
                def predict(self, data, **kwargs):
                    try:
                        response = self.client.predict(self.model_name, data)
                        return response.get('result', {})
                    except Exception as e:
                        logging.error(f"Error in model prediction: {e}")
                        # Return None on error
                        return None
            
            return ModelWrapper(self.model_client, model_name)
        else:
            # Fall back to direct MLflow
            try:
                import mlflow
                # Get latest version
                versions = self.client.search_model_versions(f"name='{model_name}'")
                
                if not versions:
                    self.logger.error(f"No versions found for model {model_name}")
                    return None
                    
                latest_version = max([int(mv.version) for mv in versions])
                model_uri = f"models:/{model_name}/{latest_version}"
                
                # Load the model directly from MLflow
                model = mlflow.pyfunc.load_model(model_uri)
                
                return model
            except Exception as e:
                self.logger.error(f"Error loading model {model_name}: {e}")
                return None