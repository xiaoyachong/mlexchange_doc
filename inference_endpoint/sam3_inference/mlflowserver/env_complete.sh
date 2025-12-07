# ===== Existing Configuration =====
# (Keep all your existing variables)
USER_NAME=user1
USER_PASSWORD=password123
MLFLOW_TRACKING_URI=http://mlflow:5000
DATA_TILED_URI=http://tiled:8000/api/v1/metadata/data
# ... etc ...

# ===== SAM3 Inference Configuration =====

# SAM3 Service Endpoint (client configuration)
# This is where your Dash app will send requests
SAM3_INFERENCE_URL=http://sam3-inference:5001/invocations

# Request timeout in seconds (default: 120)
# Increase for large images or slow GPUs
SAM3_TIMEOUT=120

# Hugging Face Token (REQUIRED)
# Get from: https://huggingface.co/settings/tokens
# Needs "Read" access to download models
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# SAM3 Model Variant (optional)
# Default: facebook/sam3
# Alternatives: facebook/sam3-hq, facebook/sam3-base
SAM_MODEL_NAME=facebook/sam3

# ===== Optional: Advanced Configuration =====

# Cache directory for model weights (inside container)
TRANSFORMERS_CACHE=/cache/transformers
HF_HOME=/cache/huggingface

# MLflow Model Registry Settings
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
