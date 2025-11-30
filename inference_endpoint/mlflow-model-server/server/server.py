import os
import time
import json
import logging
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model-server')

# Configure MLflow
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_USERNAME = os.environ.get('MLFLOW_TRACKING_USERNAME', '')
MLFLOW_PASSWORD = os.environ.get('MLFLOW_TRACKING_PASSWORD', '')
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/mlflow_cache')
PRELOAD_MODELS = os.environ.get('PRELOAD_MODELS', '').split(',')

# Set MLflow environment variables
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD

# Initialize Flask app
app = Flask(__name__)

# Create model cache directory
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Initialize MLflow client
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Model cache - keep models in memory
loaded_models = {}

def load_model(model_name, version="latest"):
    """Load a model from MLflow or memory cache"""
    cache_key = f"{model_name}/{version}"
    
    # Check if model is already loaded
    if cache_key in loaded_models:
        logger.debug(f"Using in-memory model: {cache_key}")
        return loaded_models[cache_key]
    
    try:
        # Resolve version number if "latest"
        if version == "latest":
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.error(f"No versions found for model {model_name}")
                return None
            version = max([int(mv.version) for mv in versions])
            logger.info(f"Resolved version for {model_name}: {version}")
        
        # Check if model is in disk cache
        cache_path = os.path.join(MODEL_CACHE_DIR, f"{model_name}_v{version}")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading model from disk cache: {cache_path}")
            model = mlflow.pyfunc.load_model(cache_path)
        else:
            # Download model
            logger.info(f"Downloading model {model_name} (version {version}) from MLflow")
            model_uri = f"models:/{model_name}/{version}"
            
            # Download to cache
            mlflow.artifacts.download_artifacts(model_uri, dst_path=cache_path)
            
            # Load model
            model = mlflow.pyfunc.load_model(cache_path)
        
        # Store in memory cache
        loaded_models[cache_key] = model
        logger.info(f"Model {model_name} (version {version}) loaded successfully")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None

def serialize_prediction(result):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, dict):
        return {k: serialize_prediction(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [serialize_prediction(item) for item in result]
    elif isinstance(result, (np.int32, np.int64)):
        return int(result)
    elif isinstance(result, (np.float32, np.float64)):
        return float(result)
    elif isinstance(result, np.bool_):
        return bool(result)
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(loaded_models.keys()),
        'timestamp': time.time()
    })

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    """Run prediction with specified model"""
    start_time = time.time()
    
    # Get data from request
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.json.get('data')
    if data is None:
        return jsonify({'error': 'No data provided'}), 400
    
    # Convert data to numpy array if it's a list
    if isinstance(data, list):
        try:
            data = np.array(data)
        except Exception as e:
            return jsonify({'error': f'Error converting data to numpy array: {e}'}), 400
    
    # Get model version
    version = request.args.get('version', 'latest')
    
    # Load model
    model = load_model(model_name, version)
    if model is None:
        return jsonify({'error': f'Model {model_name} (version {version}) not found'}), 404
    
    try:
        # Run prediction
        result = model.predict(data)
        
        # Convert to JSON-serializable format
        result = serialize_prediction(result)
        
        # Return result
        return jsonify({
            'result': result,
            'model': model_name,
            'version': version if version != "latest" else "latest",
            'timing_ms': round((time.time() - start_time) * 1000, 2)
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Preload models if specified
    for model_name in PRELOAD_MODELS:
        if model_name.strip():
            logger.info(f"Preloading model: {model_name}")
            load_model(model_name.strip())
    
    # Start server
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting model server on port {port}")
    app.run(host='0.0.0.0', port=port)