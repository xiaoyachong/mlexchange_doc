import logging
import os
import torch
import numpy as np
from .redis_model_store import RedisModelStore

# Environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "kvrocks")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6666))

logger = logging.getLogger("arroyo_reduction.reducer")

class LatentSpaceReducer:
    """
    Responsible for taking an image, encoding it into a
    latent space, and reducing it to 2D
    """

    def __init__(self):
        """Initialize the reducer with models from Redis"""
        # Initialize model loading status flags
        self.is_loading_model = False
        self.loading_model_type = None
        
        # Initialize Redis model store
        self.model_store = RedisModelStore(host=REDIS_HOST, port=REDIS_PORT)
        
        # Get model selections from Redis
        self.autoencoder_model_name = self.model_store.get_autoencoder_model()
        self.dimred_model_name = self.model_store.get_dimred_model()
        
        logger.info(f"Using autoencoder model: {self.autoencoder_model_name}")
        logger.info(f"Using dimension reduction model: {self.dimred_model_name}")
        
        # Check for CUDA else use CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        self.device = device
        
        # Initialize MLflow client with model server
        from src.utils.mlflow_utils import MLflowClient
        self.mlflow_client = MLflowClient()
        
        # Set loading flags before loading models
        self.is_loading_model = True
        self.loading_model_type = "initial"
        
        try:
            # Load models
            self.current_torch_model = self.mlflow_client.load_model(self.autoencoder_model_name)
            self.current_dim_reduction_model = self.mlflow_client.load_model(self.dimred_model_name)
            logger.info("Initial models loaded successfully")
        finally:
            # Reset loading flags
            self.is_loading_model = False
            self.loading_model_type = None
        
        # Subscribe to model update channel if supported
        self._subscribe_to_model_updates()

    def reduce(self, message):
        """Process an image through the models to get feature vectors"""
        
        # Check if models are currently being loaded
        if self.is_loading_model:
            logger.info(f"Waiting for {self.loading_model_type} model to finish loading...")
            # Return a placeholder while models are loading
            return np.zeros((1, 2))  # Return empty vector during loading
            
        try:
            # Get numpy array from message
            img_array = message.image.array.copy()  # Create a fresh copy
            # Ensure array is properly formatted for model input
            img_array = np.ascontiguousarray(img_array)

            # Additional debugging for the image data
            logger.info(f"Input image shape: {img_array.shape}, dtype: {img_array.dtype}. Image min: {img_array.min()}, max: {img_array.max()}")
            
        except Exception as e:
            logger.error(f"Error in image preparation: {e}")
            return np.zeros((1, 2))  # Return empty vector on error
        
        # Process with autoencoder to get latent features
        try:
            # Pass numpy array directly to model, the predict() API will handle data preprocessing 
            autoencoder_result = self.current_torch_model.predict(img_array)  
            latent_features = autoencoder_result["latent_features"]
            logger.info(f"Latent features shape: {latent_features.shape}")
            
        except Exception as e:
            logger.error(f"Error in autoencoder processing: {e}")
            return np.zeros((1, 2))  # Return empty vector on error
        
        # Apply dimension reduction directly with latent features
        try:            
            umap_result = self.current_dim_reduction_model.predict(latent_features)  
            f_vec = umap_result["umap_coords"]
            logger.info(f"Feature vector shape: {f_vec.shape}")
            return f_vec
        except Exception as e:
            logger.error(f"Error in dimension reduction: {e}")
            return np.zeros((1, 2))  # Return empty vector on error

    def _handle_model_update(self, update):
        """Handle a model update from Redis PubSub"""
        try:
            model_type = update.get("model_type")
            model_name = update.get("model_name")
            
            if not model_type or not model_name:
                logger.warning(f"Invalid model update: {update}")
                return
            
            # Check if this is a duplicate update for the same model
            if (model_type == "autoencoder" and model_name == self.autoencoder_model_name) or \
            (model_type == "dimred" and model_name == self.dimred_model_name):
                logger.info(f"Ignoring duplicate model update: {model_type} = {model_name} (already loaded)")
                return
                
            logger.info(f"Received model update: {model_type} = {model_name}")
            
            # Set loading flags before updating models
            self.is_loading_model = True
            self.loading_model_type = model_type
            
            try:
                # Update the appropriate model
                if model_type == "autoencoder":
                    logger.info(f"Loading new autoencoder model: {model_name}...")
                    self.autoencoder_model_name = model_name
                    self.current_torch_model = self.mlflow_client.load_model(model_name)
                    logger.info(f"Successfully loaded new autoencoder model: {model_name}")
                elif model_type == "dimred":
                    logger.info(f"Loading new dimension reduction model: {model_name}...")
                    self.dimred_model_name = model_name
                    self.current_dim_reduction_model = self.mlflow_client.load_model(model_name)
                    logger.info(f"Successfully loaded new dimension reduction model: {model_name}")
                else:
                    logger.warning(f"Unknown model type: {model_type}")
            finally:
                # Reset loading flags
                self.is_loading_model = False
                self.loading_model_type = None
                logger.info(f"Model update complete. Ready to process images.")
        except Exception as e:
            # Ensure we reset flags even if there's an error
            self.is_loading_model = False
            self.loading_model_type = None
            logger.error(f"Error handling model update: {e}")
            
    def _subscribe_to_model_updates(self):
        """Subscribe to model updates from Redis"""
        try:
            self.model_store.subscribe_to_model_updates(self._handle_model_update)
            logger.info("Subscribed to model updates")
        except Exception as e:
            logger.error(f"Error subscribing to model updates: {e}")