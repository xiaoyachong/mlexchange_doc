import os
import sys
import logging
import time
import argparse
import mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preload_models')

def preload_models(model_names, cache_dir, mlflow_tracking_uri, mlflow_username=None, mlflow_password=None):
    """
    Preload models into the cache directory
    
    Args:
        model_names (list): List of model names to preload
        cache_dir (str): Directory to store models
        mlflow_tracking_uri (str): MLflow tracking server URI
        mlflow_username (str): MLflow username
        mlflow_password (str): MLflow password
    """
    # Set environment variables
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    if mlflow_username:
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_username
    if mlflow_password:
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_password
    
    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"Preloading {len(model_names)} models to {cache_dir}")
    
    # Download each model
    for model_name in model_names:
        try:
            logger.info(f"Processing model: {model_name}")
            
            # Get latest version
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.warning(f"No versions found for {model_name}, skipping.")
                continue
                
            latest_version = max([int(mv.version) for mv in versions])
            logger.info(f"Latest version of {model_name} is {latest_version}")
            
            # Define cache path
            cache_path = os.path.join(cache_dir, f"{model_name}_v{latest_version}")
            
            # Skip if already cached
            if os.path.exists(cache_path):
                logger.info(f"Model {model_name} (version {latest_version}) already cached, skipping.")
                continue
            
            # Download model
            start_time = time.time()
            logger.info(f"Downloading {model_name} (version {latest_version})...")
            
            model_uri = f"models:/{model_name}/{latest_version}"
            mlflow.artifacts.download_artifacts(model_uri, dst_path=cache_path)
            
            logger.info(f"Successfully downloaded {model_name} in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preload MLflow models into cache")
    parser.add_argument("--models", type=str, required=True, 
                      help="Comma-separated list of model names to preload")
    parser.add_argument("--cache-dir", type=str, default="/tmp/mlflow_model_cache",
                      help="Directory to store cached models")
    parser.add_argument("--mlflow-uri", type=str, required=True,
                      help="MLflow tracking server URI")
    parser.add_argument("--mlflow-username", type=str, default=None,
                      help="MLflow username (if authentication is enabled)")
    parser.add_argument("--mlflow-password", type=str, default=None,
                      help="MLflow password (if authentication is enabled)")
    
    args = parser.parse_args()
    
    # Split model names
    model_list = [name.strip() for name in args.models.split(',') if name.strip()]
    
    # Preload models
    preload_models(
        model_names=model_list,
        cache_dir=args.cache_dir,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_username=args.mlflow_username,
        mlflow_password=args.mlflow_password
    )