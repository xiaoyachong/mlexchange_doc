# MLflow Model Server for Fast Inference

This repository contains a simplified MLflow model server designed specifically for providing fast inference on individual samples (images). The server implements efficient in-memory and disk caching to minimize latency and optimize resource usage.

## Features

- **Lightweight**: Minimal dependencies and focused functionality
- **Fast Inference**: Optimized for single-image processing with low latency
- **Model Caching**: Both in-memory and disk caching for optimal performance
- **Preloading**: Automatically preload common models at startup
- **Simple API**: Clean, straightforward REST API for predictions
- **Easy Integration**: Simple client library for easy integration with existing applications

## Project Structure

```
mlflow-model-server/
├── docker-compose.yml      # Docker setup
├── README.md               # This documentation
├── server/
│   ├── Dockerfile          # Server container definition
│   ├── requirements.txt    # Minimal dependencies
│   ├── server.py           # Simple server implementation
│   └── preload_models.py   # Script to pre-download models
└── client/
    └── model_client.py     # Simple client library
```

## Quick Start

### 1. Start the server

```bash
# Clone the repository
git clone https://github.com/your-username/mlflow-model-server.git
cd mlflow-model-server

# Set environment variables (if needed)
export MLFLOW_USERNAME=your_username
export MLFLOW_PASSWORD=your_password

# Start the server
docker-compose up -d
```

### 2. Use the client

```python
from client.model_client import ModelClient

# Initialize client
client = ModelClient("http://localhost:5001")

# Check health
health = client.health()
print(f"Server status: {health['status']}")
print(f"Models loaded: {health['models_loaded']}")

# Run prediction with an image
import numpy as np
# Create a sample image (512x512 RGB)
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# Get latent features from autoencoder
result = client.predict("pytorch_autoencoder_v0.0.5", image)
latent_features = np.array(result['result']['latent_features'])

# Run dimension reduction
umap_result = client.predict("umap_v1.0.0", latent_features)
umap_coords = np.array(umap_result['result']['umap_coords'])

print(f"Latent features shape: {latent_features.shape}")
print(f"UMAP coordinates shape: {umap_coords.shape}")
```

## API Reference

### Health Check

```
GET /health
```

Returns the server status, list of loaded models, and timestamp.

Example response:
```json
{
  "status": "healthy",
  "models_loaded": [
    "pytorch_autoencoder_v0.0.5/latest",
    "umap_v1.0.0/latest"
  ],
  "timestamp": 1623456789.123
}
```

### Prediction

```
POST /predict/<model_name>
```

Parameters:
- `model_name`: Name of the model to use for prediction
- `version` (query parameter, optional): Model version, defaults to "latest"

Request body:
```json
{
  "data": [...]  // Input data (array or nested arrays)
}
```

Example response:
```json
{
  "result": {
    "latent_features": [...]
  },
  "model": "pytorch_autoencoder_v0.0.5",
  "version": "latest",
  "timing_ms": 123.45
}
```

## Configuration

The server can be configured using environment variables:

- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (default: `http://mlflow:5000`)
- `MLFLOW_TRACKING_USERNAME`: MLflow authentication username (if required)
- `MLFLOW_TRACKING_PASSWORD`: MLflow authentication password (if required)
- `MODEL_CACHE_DIR`: Directory for model cache (default: `/mlflow_cache`)
- `PRELOAD_MODELS`: Comma-separated list of models to preload at startup
- `PORT`: Server port (default: `5001`)

## Integration with Latent Space Explorer

To integrate with the Latent Space Explorer application, you'll need to update your MLflowClient to use the model server. The provided `model_client.py` shows how to create a compatible interface.

## Example Integration with Existing Application

Here's how to modify your `MLflowClient` class to use the model server:

```python
class MLflowClient:
    """A wrapper class for MLflow client operations."""
    
    def __init__(
        self, 
        model_server_url=None,
        tracking_uri=None,
        username=None, 
        password=None,
    ):
        """Initialize the client with connection parameters."""
        self.model_server_url = model_server_url or os.getenv("MODEL_SERVER_URL", "http://model-server:5001")
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.username = username or os.getenv("MLFLOW_TRACKING_USERNAME", "")
        self.password = password or os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        
        # Initialize model server client
        from client.model_client import ModelClient
        self.model_client = ModelClient(self.model_server_url)
        
        # Check if model server is available
        try:
            self.model_client.health()
            self.server_available = True
        except Exception as e:
            self.server_available = False
            # Initialize direct MLflow client as fallback
            # ...

    def load_model(self, model_name):
        """Load a model from the server or direct MLflow"""
        if model_name is None:
            return None
        
        if self.server_available:
            # Create a model wrapper that uses the server
            class ModelWrapper:
                def __init__(self, client, model_name):
                    self.client = client
                    self.model_name = model_name
                
                def predict(self, data, **kwargs):
                    response = self.client.predict(self.model_name, data)
                    return response.get('result', {})
            
            return ModelWrapper(self.model_client, model_name)
        else:
            # Fall back to direct MLflow
            # ...
```

## License

MIT
