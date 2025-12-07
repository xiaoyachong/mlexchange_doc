"""
SAM3 FastAPI Inference Server
Standalone inference server compatible with MLflow client format

Start with:
    uvicorn main:app --host 0.0.0.0 --port 5001 --workers 1

Or with Docker:
    docker-compose up -d sam3-inference
"""
import logging
import os
import base64
import time
from io import BytesIO
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import Sam3Model, Sam3Processor

# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sam3_server")

# ===== Configuration =====
HF_TOKEN = os.getenv("HF_TOKEN")
SAM_MODEL_NAME = os.getenv("SAM_MODEL_NAME", "facebook/sam3")

def get_device():
    """Detect best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
logger.info(f"Device selected: {DEVICE}")

# ===== FastAPI App =====
app = FastAPI(
    title="SAM3 Inference Server",
    description="REST API for SAM3 image segmentation with bounding box prompts",
    version="1.0.0"
)

# ===== Global Model Variables =====
model = None
processor = None
model_load_time = None


# ===== Request/Response Models =====
class InstanceRequest(BaseModel):
    """Single inference instance"""
    image: str = Field(..., description="Base64 encoded PNG/JPEG image")
    boxes: List[List[float]] = Field(..., description="Bounding boxes [[x1,y1,x2,y2],...]")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Mask binarization threshold")


class InferenceRequest(BaseModel):
    """MLflow-compatible request format"""
    instances: List[InstanceRequest]


class PredictionResponse(BaseModel):
    """Single prediction response"""
    masks: List[str] = Field(..., description="Base64 encoded mask images")
    scores: List[float] = Field(..., description="Confidence scores")
    num_masks: int = Field(..., description="Number of masks generated")
    success: bool = Field(..., description="Whether inference succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    inference_time: Optional[float] = Field(None, description="Processing time in seconds")


class InferenceResponse(BaseModel):
    """MLflow-compatible response format"""
    predictions: List[PredictionResponse]


# ===== Startup Event =====
@app.on_event("startup")
async def load_model():
    """Load SAM3 model on server startup"""
    global model, processor, model_load_time
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"Loading SAM3 Model: {SAM_MODEL_NAME}")
    logger.info(f"Target Device: {DEVICE}")
    logger.info("=" * 80)
    
    try:
        # Authenticate with Hugging Face
        if HF_TOKEN:
            from huggingface_hub import login
            login(token=HF_TOKEN)
            logger.info("✓ Authenticated with Hugging Face")
        else:
            logger.warning("⚠ No HF_TOKEN provided, may fail for gated models")
        
        # Load model and processor
        logger.info("Loading SAM3 model (this may take 30-60 seconds)...")
        model = Sam3Model.from_pretrained(SAM_MODEL_NAME).to(DEVICE)
        processor = Sam3Processor.from_pretrained(SAM_MODEL_NAME)
        
        model_load_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✓ SAM3 model loaded successfully in {model_load_time:.1f}s")
        logger.info(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ Failed to load model: {e}")
        logger.error("=" * 80)
        raise RuntimeError(f"Model loading failed: {e}")


# ===== Helper Functions =====
def decode_image(image_b64: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")


def encode_mask(mask: torch.Tensor) -> str:
    """Encode mask tensor to base64 string"""
    try:
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np, mode='L')
        buffer = BytesIO()
        mask_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Mask encoding failed: {e}")


def run_inference(
    image: Image.Image,
    boxes: List[List[float]],
    threshold: float,
    mask_threshold: float
) -> dict:
    """
    Run SAM3 inference on image with bounding boxes
    
    Returns:
        dict with 'masks' (tensors) and 'scores' (list)
    """
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")
    
    # Validate boxes
    if not boxes or len(boxes) == 0:
        raise ValueError("At least one bounding box required")
    
    # Prepare inputs
    box_labels = [1] * len(boxes)  # All positive prompts
    
    inputs = processor(
        images=image,
        input_boxes=[boxes],
        input_boxes_labels=[box_labels],
        return_tensors="pt"
    ).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    return results


# ===== API Endpoints =====
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SAM3 Inference Server",
        "version": "1.0.0",
        "model": SAM_MODEL_NAME,
        "device": DEVICE,
        "status": "ready" if model is not None else "loading",
        "model_load_time": f"{model_load_time:.1f}s" if model_load_time else None
    }


@app.get("/health", tags=["Info"], status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet"
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model": SAM_MODEL_NAME
    }


@app.post(
    "/invocations",
    response_model=InferenceResponse,
    tags=["Inference"],
    status_code=status.HTTP_200_OK
)
async def segment_invocations(request: InferenceRequest):
    """
    Main inference endpoint (MLflow-compatible format)
    
    Request format:
    ```json
    {
        "instances": [{
            "image": "base64_encoded_image",
            "boxes": [[x1, y1, x2, y2], ...],
            "threshold": 0.5,
            "mask_threshold": 0.5
        }]
    }
    ```
    
    Response format:
    ```json
    {
        "predictions": [{
            "masks": ["base64_mask1", "base64_mask2"],
            "scores": [0.98, 0.95],
            "num_masks": 2,
            "success": true,
            "inference_time": 5.2
        }]
    }
    ```
    """
    if model is None or processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    predictions = []
    
    for idx, instance in enumerate(request.instances):
        inference_start = time.time()
        
        try:
            logger.info(f"Processing instance {idx+1}/{len(request.instances)}")
            logger.info(f"  Boxes: {len(instance.boxes)}")
            
            # Decode image
            image = decode_image(instance.image)
            logger.info(f"  Image size: {image.size}")
            
            # Run inference
            results = run_inference(
                image=image,
                boxes=instance.boxes,
                threshold=instance.threshold,
                mask_threshold=instance.mask_threshold
            )
            
            # Encode masks
            encoded_masks = []
            scores = []
            
            if "masks" in results and len(results["masks"]) > 0:
                for mask in results["masks"]:
                    encoded_mask = encode_mask(mask)
                    encoded_masks.append(encoded_mask)
                
                if "scores" in results:
                    scores = results["scores"].cpu().tolist()
                
                inference_time = time.time() - inference_start
                logger.info(f"✓ Generated {len(encoded_masks)} masks in {inference_time:.1f}s")
                
                predictions.append(PredictionResponse(
                    masks=encoded_masks,
                    scores=scores,
                    num_masks=len(encoded_masks),
                    success=True,
                    error=None,
                    inference_time=inference_time
                ))
            else:
                inference_time = time.time() - inference_start
                logger.warning(f"No masks generated for instance {idx+1}")
                
                predictions.append(PredictionResponse(
                    masks=[],
                    scores=[],
                    num_masks=0,
                    success=True,
                    error="No masks generated (try adjusting thresholds)",
                    inference_time=inference_time
                ))
            
        except ValueError as e:
            inference_time = time.time() - inference_start
            logger.error(f"Validation error for instance {idx+1}: {e}")
            
            predictions.append(PredictionResponse(
                masks=[],
                scores=[],
                num_masks=0,
                success=False,
                error=str(e),
                inference_time=inference_time
            ))
            
        except Exception as e:
            inference_time = time.time() - inference_start
            logger.error(f"Inference error for instance {idx+1}: {e}", exc_info=True)
            
            predictions.append(PredictionResponse(
                masks=[],
                scores=[],
                num_masks=0,
                success=False,
                error=f"Inference failed: {str(e)}",
                inference_time=inference_time
            ))
    
    return InferenceResponse(predictions=predictions)


@app.post("/segment", tags=["Inference"], status_code=status.HTTP_200_OK)
async def segment_simple(
    image: str,
    boxes: List[List[float]],
    threshold: float = 0.5,
    mask_threshold: float = 0.5
):
    """
    Simplified inference endpoint (single image)
    
    Use this for simpler clients that don't need MLflow format
    """
    request = InferenceRequest(instances=[
        InstanceRequest(
            image=image,
            boxes=boxes,
            threshold=threshold,
            mask_threshold=mask_threshold
        )
    ])
    
    response = await segment_invocations(request)
    return response.predictions[0]


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Basic metrics endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB",
        }
    
    return {
        "model": SAM_MODEL_NAME,
        "device": DEVICE,
        "model_loaded": model is not None,
        "model_load_time": model_load_time,
        **gpu_info
    }


# ===== Error Handlers =====
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# ===== Main Entry Point =====
if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        workers=1,  # Use 1 worker with GPU
        log_level="info",
        access_log=True
    )
