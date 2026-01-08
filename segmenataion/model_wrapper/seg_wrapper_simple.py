#!/usr/bin/env python
"""
Segmentation model wrapper for MLflow.
Provides functionality to load segmentation models and register them with MLflow.
"""

import os
import time
import traceback
from datetime import datetime

import mlflow
import numpy as np
import torch
import torch.nn as nn


def get_file_size_mb(filepath):
    """Get file size in MB"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)


class SegmentationModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for segmentation models with direct model access
    """

    def __init__(
        self,
        network_type="DLSIA TUNet",
        num_classes=2,
        in_channels=1,
        image_shape=(64, 64),
        **network_params
    ):
        self.model = None
        self.network_type = network_type
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.image_shape = tuple(image_shape)
        self.network_params = network_params
        self.device = None

    def load_context(self, context):
        """Load segmentation model from context artifacts"""
        
        # Check for CUDA availability and set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        # Get model weights path
        weights_path = context.artifacts.get("weights_path")
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

        print(
            f"Initializing {self.network_type} with num_classes={self.num_classes}, "
            f"image_shape={self.image_shape}"
        )

        # Build the network architecture from dlsia package
        self.model = self._build_network()

        # Load weights
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            
            # Handle different save formats
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                # Checkpoint format
                self.model.load_state_dict(state_dict['model_state_dict'])
            elif isinstance(state_dict, dict):
                # Direct state dict
                self.model.load_state_dict(state_dict)
            else:
                raise ValueError(f"Unexpected format in weights file: {type(state_dict)}")
            
            print("✅ Model weights loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading model weights: {e}")
            traceback.print_exc()
            raise

        # Set model to eval mode and move to device
        self.model.eval()
        self.model = self.model.to(self.device)

    def _build_network(self):
        """Build the network architecture based on network_type"""
        if self.network_type == "DLSIA MSDNet":
            return self._build_msdnet()
        elif self.network_type == "DLSIA TUNet":
            return self._build_tunet()
        elif self.network_type == "DLSIA TUNet3+":
            return self._build_tunet3plus()
        else:
            raise ValueError(f"Unsupported network type: {self.network_type}")

    def _build_msdnet(self):
        """Build MSDNet architecture"""
        from dlsia.core.networks import msdnet
        
        # Get activation, normalization, and convolution
        activation = getattr(nn, self.network_params.get('activation', 'ReLU'))()
        normalization = getattr(nn, self.network_params.get('normalization', 'BatchNorm2d'))
        convolution = getattr(nn, self.network_params.get('convolution', 'Conv2d'))
        
        network = msdnet.MixedScaleDenseNetwork(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            num_layers=self.network_params.get('num_layers', 3),
            layer_width=self.network_params.get('layer_width', 1),
            max_dilation=self.network_params.get('max_dilation', 5),
            activation=activation,
            normalization=normalization,
            convolution=convolution,
        )
        return network

    def _build_tunet(self):
        """Build TUNet architecture"""
        from dlsia.core.networks import tunet
        
        activation = getattr(nn, self.network_params.get('activation', 'ReLU'))()
        normalization = getattr(nn, self.network_params.get('normalization', 'BatchNorm2d'))
        
        network = tunet.TUNet(
            image_shape=self.image_shape,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            depth=self.network_params.get('depth', 4),
            base_channels=self.network_params.get('base_channels', 32),
            growth_rate=self.network_params.get('growth_rate', 2),
            hidden_rate=self.network_params.get('hidden_rate', 1),
            activation=activation,
            normalization=normalization,
        )
        return network

    def _build_tunet3plus(self):
        """Build TUNet3+ architecture"""
        from dlsia.core.networks import tunet3plus
        
        activation = getattr(nn, self.network_params.get('activation', 'ReLU'))()
        normalization = getattr(nn, self.network_params.get('normalization', 'BatchNorm2d'))
        
        network = tunet3plus.TUNet3Plus(
            image_shape=self.image_shape,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            depth=self.network_params.get('depth', 4),
            base_channels=self.network_params.get('base_channels', 32),
            growth_rate=self.network_params.get('growth_rate', 2),
            hidden_rate=self.network_params.get('hidden_rate', 1),
            carryover_channels=self.network_params.get('carryover_channels', 32),
            activation=activation,
            normalization=normalization,
        )
        return network

    def predict(self, context, model_input):
        """
        Standard predict method for segmentation model

        Args:
            context: MLflow context
            model_input: Input data as numpy array (N, C, H, W) or (N, H, W)

        Returns:
            Dictionary with segmentation predictions and probabilities
        """
        if self.model is None:
            raise RuntimeError("Segmentation model not loaded. Call load_context first.")

        # Validate input
        if not isinstance(model_input, np.ndarray):
            raise ValueError(f"Input must be a numpy array, got {type(model_input)}")

        # Handle different input shapes
        if len(model_input.shape) == 3:
            # (N, H, W) -> (N, C, H, W)
            model_input = np.expand_dims(model_input, axis=1)
        elif len(model_input.shape) == 2:
            # (H, W) -> (N, C, H, W)
            model_input = np.expand_dims(np.expand_dims(model_input, axis=0), axis=0)

        # Validate shape
        if len(model_input.shape) != 4:
            raise ValueError(
                f"Input must be 4D array (N, C, H, W), got shape {model_input.shape}"
            )

        # Convert to tensor and move to device
        input_tensor = torch.tensor(model_input, dtype=torch.float32).to(self.device)

        # Process with model
        with torch.no_grad():
            # Forward pass
            output = self.model(input_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get predictions (argmax)
            predictions = torch.argmax(probabilities, dim=1)

            # Convert to numpy
            predictions_np = predictions.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()

        # Return results
        return {
            "predictions": predictions_np,
            "probabilities": probabilities_np,
        }


def save_segmentation_model_with_wrapper(
    model_config, tracking_uri, experiment_name, model_name=None
):
    """
    Save segmentation model using PyFunc wrapper

    Args:
        model_config: Dictionary with model configuration
        tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        model_name: Optional model name, defaults to name from config with date

    Returns:
        Tuple of (model_name, run_id) or (None, None) on failure
    """

    # Set MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Set model name
    if model_name is None:
        model_name = f"{model_config['name']}_v{datetime.now().strftime('%Y%m%d')}"

    print(f"\nSaving segmentation model with PyFunc wrapper as: {model_name}")

    start_time = time.time()

    with mlflow.start_run(
        run_name=f"seg_model_wrapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        print(f"Run ID: {run.info.run_id}")
        print(f"MLflow tracking URI: {tracking_uri}")

        # Check file size and existence
        if not os.path.exists(model_config["weights_path"]):
            print(f"❌ Error: Weights file not found at {model_config['weights_path']}")
            return None, None

        weights_size = get_file_size_mb(model_config["weights_path"])
        print(f"\nWeights file size: {weights_size:.1f} MB")

        try:
            # Get model parameters
            num_classes = model_config.get("num_classes", 2)
            in_channels = model_config.get("in_channels", 1)
            image_shape = model_config.get("image_shape", (64, 64))
            network_type = model_config.get("network_type", "DLSIA TUNet")

            # Create model wrapper
            seg_wrapper = SegmentationModelWrapper(
                network_type=network_type,
                num_classes=num_classes,
                in_channels=in_channels,
                image_shape=image_shape,
                **model_config.get("network_params", {})
            )

            # Log model information
            mlflow.log_params(
                {
                    "model_name": model_config["name"],
                    "network_type": network_type,
                    "num_classes": num_classes,
                    "in_channels": in_channels,
                    "image_shape": f"{image_shape[0]}x{image_shape[1]}",
                    "weights_size_mb": weights_size,
                    "using_wrapper": True,
                }
            )

            # Set tags
            mlflow.set_tags({"exp_type": "segmentation", "model_type": "segmentation"})

            # Create artifacts dictionary
            artifacts = {
                "weights_path": model_config["weights_path"],
            }

            # Define explicit requirements
            pip_requirements = [
                "torch==2.2.2",
                "numpy",
                "mlflow==2.22.0",
                "dlsia",
            ]

            # Log the segmentation model with PyFunc wrapper
            print("\nLogging segmentation model with PyFunc wrapper to MLflow...")
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=seg_wrapper,
                artifacts=artifacts,
                registered_model_name=model_name,
                pip_requirements=pip_requirements,
                code_path=[__file__],
            )

            # Log timing
            total_time = time.time() - start_time
            mlflow.log_metric("upload_time_seconds", total_time)

            print(
                f"\n✅ Segmentation model saved with PyFunc wrapper in {total_time:.1f}s!"
            )
            print(f"Model name: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"MLflow UI: {tracking_uri}")

            return model_name, run.info.run_id

        except Exception as e:
            print(f"\n❌ Error saving segmentation model with wrapper: {e}")
            traceback.print_exc()
            return None, None
