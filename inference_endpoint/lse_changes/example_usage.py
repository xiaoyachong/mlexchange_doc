"""
Example script showing how to use the model server with the Latent Space Explorer application
"""

import numpy as np
from client.model_client import ModelClient
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('example')

def main():
    # Initialize model client
    server_url = "http://localhost:5001"
    client = ModelClient(server_url)
    
    # Check server health
    try:
        health = client.health()
        logger.info(f"Server status: {health['status']}")
        logger.info(f"Models loaded: {health['models_loaded']}")
    except Exception as e:
        logger.error(f"Error connecting to model server: {e}")
        logger.info("Make sure the model server is running")
        return
    
    # Create sample image (512x512 RGB)
    logger.info("Creating sample image")
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Run autoencoder
    logger.info("Running autoencoder prediction")
    start_time = time.time()
    try:
        result = client.predict("pytorch_autoencoder_v0.0.5", image)
        latent_features = np.array(result['result']['latent_features'])
        logger.info(f"Autoencoder prediction time: {time.time() - start_time:.2f}s")
        logger.info(f"Latent features shape: {latent_features.shape}")
    except Exception as e:
        logger.error(f"Error running autoencoder prediction: {e}")
        return
    
    # Run dimension reduction
    logger.info("Running dimension reduction")
    start_time = time.time()
    try:
        umap_result = client.predict("umap_v1.0.0", latent_features)
        umap_coords = np.array(umap_result['result']['umap_coords'])
        logger.info(f"Dimension reduction time: {time.time() - start_time:.2f}s")
        logger.info(f"UMAP coordinates shape: {umap_coords.shape}")
    except Exception as e:
        logger.error(f"Error running dimension reduction: {e}")
        return
    
    # Run predictions in sequence for throughput test
    logger.info("Running throughput test with 10 images")
    total_time = 0
    num_images = 10
    
    for i in range(num_images):
        # Create random image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Run full pipeline
        start_time = time.time()
        
        # Autoencoder
        ae_result = client.predict("pytorch_autoencoder_v0.0.5", test_image)
        features = np.array(ae_result['result']['latent_features'])
        
        # Dimension reduction
        dr_result = client.predict("umap_v1.0.0", features)
        coords = np.array(dr_result['result']['umap_coords'])
        
        elapsed = time.time() - start_time
        total_time += elapsed
        logger.info(f"Image {i+1}/{num_images} processed in {elapsed:.2f}s")
    
    # Print throughput stats
    avg_time = total_time / num_images
    throughput = num_images / total_time
    logger.info(f"Average processing time: {avg_time:.2f}s per image")
    logger.info(f"Throughput: {throughput:.2f} images per second")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()