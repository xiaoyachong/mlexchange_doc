#!/usr/bin/env python3
"""
Test script for SAM3 MLflow inference server
Tests the complete pipeline: encode → inference → decode
"""
import requests
import base64
import time
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

# Configuration
SAM3_URL = "http://localhost:5001/invocations"
HEALTH_URL = "http://localhost:5001/health"

print("=" * 80)
print("SAM3 Inference Server Test")
print("=" * 80)

# Step 1: Health Check
print("\n[1/4] Checking server health...")
try:
    response = requests.get(HEALTH_URL, timeout=5)
    if response.status_code == 200:
        print("✓ Server is healthy")
    else:
        print(f"✗ Server returned status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"✗ Cannot reach server: {e}")
    print("  Make sure service is running: docker-compose ps sam3-inference")
    exit(1)

# Step 2: Create test image
print("\n[2/4] Creating test image...")
img = Image.new('RGB', (512, 512), color='white')
draw = ImageDraw.Draw(img)

# Draw a red square
draw.rectangle([100, 100, 300, 300], fill='red', outline='black', width=3)
print("✓ Test image created (512x512 with red square)")

# Step 3: Encode image
print("\n[3/4] Encoding image to base64...")
buffer = BytesIO()
img.save(buffer, format='PNG')
img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
print(f"✓ Image encoded ({len(img_b64)} characters)")

# Step 4: Send inference request
print("\n[4/4] Sending inference request...")
payload = {
    "instances": [{
        "image": img_b64,
        "boxes": [[100, 100, 300, 300]],  # Box around red square
        "threshold": 0.5,
        "mask_threshold": 0.5
    }]
}

print(f"  Bounding box: [100, 100, 300, 300]")
print(f"  Thresholds: 0.5 / 0.5")
print("  Waiting for response (may take 5-30 seconds)...")

start_time = time.time()
try:
    response = requests.post(SAM3_URL, json=payload, timeout=60)
    elapsed = time.time() - start_time
    
    if response.status_code != 200:
        print(f"✗ Server returned status {response.status_code}")
        print(f"  Response: {response.text}")
        exit(1)
    
    result = response.json()
    
    # Parse response
    if "predictions" in result:
        prediction = result["predictions"][0]
        
        print(f"✓ Response received in {elapsed:.1f}s")
        print(f"  Success: {prediction.get('success', False)}")
        print(f"  Masks: {prediction.get('num_masks', 0)}")
        
        if prediction.get("scores"):
            print(f"  Scores: {[f'{s:.3f}' for s in prediction['scores']]}")
        
        if prediction.get("error"):
            print(f"  Error: {prediction['error']}")
        
        # Decode first mask
        if prediction.get("masks") and len(prediction["masks"]) > 0:
            print("\n[BONUS] Decoding mask...")
            mask_bytes = base64.b64decode(prediction["masks"][0])
            mask_img = Image.open(BytesIO(mask_bytes))
            mask_np = np.array(mask_img)
            
            print(f"  Mask shape: {mask_np.shape}")
            print(f"  Mask dtype: {mask_np.dtype}")
            print(f"  True pixels: {(mask_np > 0).sum()}")
            print(f"  Coverage: {(mask_np > 0).sum() / mask_np.size * 100:.1f}%")
            
            # Save mask
            mask_img.save("test_output_mask.png")
            print("  ✓ Saved to: test_output_mask.png")
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        
    else:
        print(f"✗ Invalid response format: {result}")
        exit(1)
        
except requests.Timeout:
    elapsed = time.time() - start_time
    print(f"✗ Request timeout after {elapsed:.1f}s")
    print("  Increase timeout or check GPU availability")
    exit(1)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
