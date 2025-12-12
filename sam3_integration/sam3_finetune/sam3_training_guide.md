# Complete SAM3 Training Data Preparation Guide

**Dataset:** N images with corresponding mask images  
**Classes:** Sand (1), Air (2), Background (3)  
**Goal:** Generate COCO JSON for both instance and semantic segmentation

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Structure](#dataset-structure)
3. [Installation](#installation)
4. [Instance Segmentation JSON](#instance-segmentation-json)
5. [Semantic Segmentation JSON](#semantic-segmentation-json)
6. [Complete Conversion Scripts](#complete-conversion-scripts)
7. [Verification](#verification)
8. [JSON Format Examples](#json-format-examples)
9. [Training Configuration](#training-configuration)
10. [Quick Start Guide](#quick-start-guide)
11. [Summary Comparison](#summary-comparison)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Libraries

```bash
pip install numpy
pip install Pillow
pip install pycocotools
pip install tqdm
pip install scipy
pip install matplotlib
```

### Your Dataset Format

```
your_dataset/
├── images/
│   ├── image_001.jpg  (H × W × 3) RGB image
│   ├── image_002.jpg
│   ├── image_003.jpg
│   └── ... (N images)
└── masks/
    ├── image_001.png  (H × W) grayscale
    ├── image_002.png
    ├── image_003.png
    └── ... (N masks)
```

**Mask pixel values:**
- `0` = unlabeled/ignore (optional)
- `1` = sand
- `2` = air  
- `3` = background

---

## Dataset Structure

### Recommended Final Structure

```
dataset/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── masks/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── _annotations.coco.json  ← Generated
├── valid/
│   ├── images/
│   ├── masks/
│   └── _annotations.coco.json  ← Generated
└── test/
    ├── images/
    ├── masks/
    └── _annotations.coco.json  ← Generated
```

---

## Installation

Create a Python script directory:

```bash
mkdir sam3_data_prep
cd sam3_data_prep
```

---

## Instance Segmentation JSON

### What is Instance Segmentation?

Each separate object gets its own annotation.

**Example:** If an image has 3 sand piles, 2 air regions, 1 background:
- 6 annotations total (one per object)
- Each object tracked separately
- Can count instances

### JSON Structure (Instance)

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "height": 1024,
      "width": 1024
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [50, 600, 300, 200],
      "area": 58000,
      "segmentation": {
        "counts": "RLE_ENCODED_STRING",
        "size": [1024, 1024]
      },
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 1,
      "bbox": [400, 650, 250, 180],
      "area": 43000,
      "segmentation": {
        "counts": "RLE_ENCODED_STRING",
        "size": [1024, 1024]
      },
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "sand",
      "supercategory": "stuff",
      "isthing": 1
    }
  ]
}
```

**Key:** `"iscrowd": 0` = Instance segmentation

---

## Semantic Segmentation JSON

### What is Semantic Segmentation?

All objects of the same class are merged into one annotation.

**Example:** If an image has 3 sand piles, 2 air regions, 1 background:
- 3 annotations total (one per class)
- All sand piles merged
- Cannot count instances

### JSON Structure (Semantic)

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "height": 1024,
      "width": 1024
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [50, 600, 930, 200],
      "area": 152000,
      "segmentation": {
        "counts": "RLE_ENCODED_STRING_ALL_SAND",
        "size": [1024, 1024]
      },
      "iscrowd": 1
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "sand",
      "supercategory": "stuff",
      "isthing": 0
    }
  ]
}
```

**Key:** `"iscrowd": 1` = Semantic segmentation

---

## Complete Conversion Scripts

### Script 1: Instance Segmentation Converter

**File:** `convert_to_instance_segmentation.py`

```python
"""
Instance Segmentation COCO JSON Generator
Separates each connected component as a different instance
"""

import numpy as np
from pycocotools import mask as mask_utils
import json
from PIL import Image
import os
from tqdm import tqdm
from scipy import ndimage

def find_connected_components(binary_mask):
    """
    Find separate connected components in a binary mask
    
    Args:
        binary_mask: H×W array with 0s and 1s
    
    Returns:
        labeled_mask: H×W array where each component has unique ID
        num_components: Number of separate components found
    """
    labeled_mask, num_components = ndimage.label(binary_mask)
    return labeled_mask, num_components


def create_instance_coco_json(
    images_dir,
    masks_dir,
    output_json_path,
    class_mapping
):
    """
    Convert mask images to COCO JSON format (Instance Segmentation)
    
    Args:
        images_dir: Directory with original images
        masks_dir: Directory with mask images (pixel value = class ID)
        output_json_path: Output JSON file path
        class_mapping: Dict mapping class IDs to names
                       e.g., {0: "background", 1: "sand", 2: "air"}
    """
    
    coco_data = {
        "info": {
            "description": "Instance Segmentation Dataset",
            "version": "1.0",
            "year": 2024
        },
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ]
    }
    
    # Create categories from class_mapping
    for class_id, class_name in class_mapping.items():
        if class_id == 0:
            continue  # Skip background or include if needed
        coco_data["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "stuff",
            "isthing": 1  # Instance segmentation
        })
    
    annotation_id = 1
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"Processing {len(image_files)} images for INSTANCE segmentation...")
    
    for img_id, img_filename in enumerate(tqdm(image_files), start=1):
        # Load original image
        img_path = os.path.join(images_dir, img_filename)
        img = Image.open(img_path)
        width, height = img.size
        
        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "height": height,
            "width": width,
            "license": 1
        })
        
        # Load mask image
        mask_filename = img_filename.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(masks_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_filename}")
            continue
        
        mask_img = np.array(Image.open(mask_path))
        
        # Process each class
        for class_id, class_name in class_mapping.items():
            if class_id == 0:
                continue  # Skip background
            
            # Extract binary mask for this class
            class_mask = (mask_img == class_id).astype(np.uint8)
            
            if not np.any(class_mask):
                continue  # Class not present in this image
            
            # Find separate connected components (instances)
            labeled_mask, num_instances = find_connected_components(class_mask)
            
            # Create annotation for each instance
            for instance_id in range(1, num_instances + 1):
                # Extract this instance only
                instance_mask = (labeled_mask == instance_id).astype(np.uint8)
                
                # Skip very small instances (noise)
                if np.sum(instance_mask) < 10:
                    continue
                
                # Convert to RLE
                rle = mask_utils.encode(np.asfortranarray(instance_mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                
                # Calculate bbox
                y_indices, x_indices = np.where(instance_mask > 0)
                x_min, x_max = int(x_indices.min()), int(x_indices.max())
                y_min, y_max = int(y_indices.min()), int(y_indices.max())
                bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                
                # Calculate area
                area = int(np.sum(instance_mask))
                
                # Add annotation
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": rle,
                    "iscrowd": 0  # INSTANCE SEGMENTATION
                })
                annotation_id += 1
    
    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ Instance Segmentation JSON created:")
    print(f"  - Output: {output_json_path}")
    print(f"  - Images: {len(coco_data['images'])}")
    print(f"  - Annotations: {len(coco_data['annotations'])}")
    print(f"  - Categories: {len(coco_data['categories'])}")
    
    # Statistics
    stats = {}
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        stats[cat_id] = stats.get(cat_id, 0) + 1
    
    print(f"\n  Instance counts per class:")
    for cat in coco_data["categories"]:
        count = stats.get(cat["id"], 0)
        print(f"    - {cat['name']}: {count} instances")


if __name__ == "__main__":
    # Configuration
    class_mapping = {
        0: "background",  # Will be skipped
        1: "sand",
        2: "air",
        3: "background"
    }
    
    # Convert train set
    create_instance_coco_json(
        images_dir="dataset/train/images",
        masks_dir="dataset/train/masks",
        output_json_path="dataset/train/_annotations_instance.coco.json",
        class_mapping=class_mapping
    )
    
    # Convert validation set
    create_instance_coco_json(
        images_dir="dataset/valid/images",
        masks_dir="dataset/valid/masks",
        output_json_path="dataset/valid/_annotations_instance.coco.json",
        class_mapping=class_mapping
    )
    
    # Convert test set
    create_instance_coco_json(
        images_dir="dataset/test/images",
        masks_dir="dataset/test/masks",
        output_json_path="dataset/test/_annotations_instance.coco.json",
        class_mapping=class_mapping
    )
```

---

### Script 2: Semantic Segmentation Converter

**File:** `convert_to_semantic_segmentation.py`

```python
"""
Semantic Segmentation COCO JSON Generator
Merges all instances of the same class into one annotation
"""

import numpy as np
from pycocotools import mask as mask_utils
import json
from PIL import Image
import os
from tqdm import tqdm


def create_semantic_coco_json(
    images_dir,
    masks_dir,
    output_json_path,
    class_mapping
):
    """
    Convert mask images to COCO JSON format (Semantic Segmentation)
    
    Args:
        images_dir: Directory with original images
        masks_dir: Directory with mask images (pixel value = class ID)
        output_json_path: Output JSON file path
        class_mapping: Dict mapping class IDs to names
                       e.g., {0: "background", 1: "sand", 2: "air"}
    """
    
    coco_data = {
        "info": {
            "description": "Semantic Segmentation Dataset",
            "version": "1.0",
            "year": 2024
        },
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ]
    }
    
    # Create categories from class_mapping
    for class_id, class_name in class_mapping.items():
        if class_id == 0:
            continue  # Skip background or include if needed
        coco_data["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "stuff",
            "isthing": 0  # Semantic segmentation
        })
    
    annotation_id = 1
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"Processing {len(image_files)} images for SEMANTIC segmentation...")
    
    for img_id, img_filename in enumerate(tqdm(image_files), start=1):
        # Load original image
        img_path = os.path.join(images_dir, img_filename)
        img = Image.open(img_path)
        width, height = img.size
        
        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "height": height,
            "width": width,
            "license": 1
        })
        
        # Load mask image
        mask_filename = img_filename.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(masks_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_filename}")
            continue
        
        mask_img = np.array(Image.open(mask_path))
        
        # Process each class (ALL instances merged)
        for class_id, class_name in class_mapping.items():
            if class_id == 0:
                continue  # Skip background
            
            # Extract binary mask for ALL instances of this class
            binary_mask = (mask_img == class_id).astype(np.uint8)
            
            # Skip if class not present
            if not np.any(binary_mask):
                continue
            
            # Convert to RLE
            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            
            # Calculate bbox (covers ALL instances)
            y_indices, x_indices = np.where(binary_mask > 0)
            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
            
            # Calculate total area (ALL instances)
            area = int(np.sum(binary_mask))
            
            # Add annotation
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": bbox,
                "area": area,
                "segmentation": rle,
                "iscrowd": 1  # SEMANTIC SEGMENTATION
            })
            annotation_id += 1
    
    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ Semantic Segmentation JSON created:")
    print(f"  - Output: {output_json_path}")
    print(f"  - Images: {len(coco_data['images'])}")
    print(f"  - Annotations: {len(coco_data['annotations'])}")
    print(f"  - Categories: {len(coco_data['categories'])}")
    
    # Statistics
    stats = {}
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        stats[cat_id] = stats.get(cat_id, 0) + 1
    
    print(f"\n  Annotations per class:")
    for cat in coco_data["categories"]:
        count = stats.get(cat["id"], 0)
        print(f"    - {cat['name']}: {count} annotations ({count} images)")


if __name__ == "__main__":
    # Configuration
    class_mapping = {
        0: "background",  # Will be skipped
        1: "sand",
        2: "air",
        3: "background"
    }
    
    # Convert train set
    create_semantic_coco_json(
        images_dir="dataset/train/images",
        masks_dir="dataset/train/masks",
        output_json_path="dataset/train/_annotations_semantic.coco.json",
        class_mapping=class_mapping
    )
    
    # Convert validation set
    create_semantic_coco_json(
        images_dir="dataset/valid/images",
        masks_dir="dataset/valid/masks",
        output_json_path="dataset/valid/_annotations_semantic.coco.json",
        class_mapping=class_mapping
    )
    
    # Convert test set
    create_semantic_coco_json(
        images_dir="dataset/test/images",
        masks_dir="dataset/test/masks",
        output_json_path="dataset/test/_annotations_semantic.coco.json",
        class_mapping=class_mapping
    )
```

---

## Verification

### Script 3: Verify JSON and Visualize

**File:** `verify_coco_json.py`

```python
"""
Verify COCO JSON and visualize masks
"""

import numpy as np
from pycocotools import mask as mask_utils
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os


def verify_coco_json(json_path, images_dir, masks_dir, num_samples=3):
    """
    Verify COCO JSON and visualize samples
    
    Args:
        json_path: Path to COCO JSON file
        images_dir: Directory with original images
        masks_dir: Directory with original mask images
        num_samples: Number of samples to visualize
    """
    print(f"\n{'='*60}")
    print(f"VERIFYING: {json_path}")
    print(f"{'='*60}\n")
    
    # Load JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Basic statistics
    print(f"Dataset Statistics:")
    print(f"  - Images: {len(coco_data['images'])}")
    print(f"  - Annotations: {len(coco_data['annotations'])}")
    print(f"  - Categories: {len(coco_data['categories'])}")
    
    # Category breakdown
    cat_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    ann_per_cat = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        ann_per_cat[cat_id] = ann_per_cat.get(cat_id, 0) + 1
    
    print(f"\n  Annotations per category:")
    for cat_id, count in ann_per_cat.items():
        print(f"    - {cat_dict[cat_id]}: {count}")
    
    # Check iscrowd
    iscrowd_vals = [ann['iscrowd'] for ann in coco_data['annotations']]
    if all(v == 0 for v in iscrowd_vals):
        seg_type = "INSTANCE SEGMENTATION"
    elif all(v == 1 for v in iscrowd_vals):
        seg_type = "SEMANTIC SEGMENTATION"
    else:
        seg_type = "MIXED"
    
    print(f"\n  Segmentation type: {seg_type}")
    
    # Verify a few samples
    print(f"\n{'='*60}")
    print(f"VERIFYING SAMPLES")
    print(f"{'='*60}\n")
    
    for i in range(min(num_samples, len(coco_data['images']))):
        img_info = coco_data['images'][i]
        img_id = img_info['id']
        img_filename = img_info['file_name']
        
        print(f"\nSample {i+1}: {img_filename}")
        
        # Load original image
        img_path = os.path.join(images_dir, img_filename)
        img = Image.open(img_path).convert('RGB')
        
        # Load original mask
        mask_filename = img_filename.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(masks_dir, mask_filename)
        original_mask = np.array(Image.open(mask_path))
        
        # Get annotations for this image
        anns = [a for a in coco_data['annotations'] if a['image_id'] == img_id]
        print(f"  - Annotations: {len(anns)}")
        
        # Decode and verify each annotation
        for ann in anns:
            cat_name = cat_dict[ann['category_id']]
            
            # Decode RLE
            rle = ann['segmentation']
            if isinstance(rle['counts'], str):
                rle['counts'] = rle['counts'].encode('utf-8')
            decoded_mask = mask_utils.decode(rle)
            
            # Compare with original
            class_id = ann['category_id']
            original_class_mask = (original_mask == class_id).astype(np.uint8)
            
            if seg_type == "INSTANCE SEGMENTATION":
                # For instance, just check subset
                overlap = np.sum(decoded_mask * original_class_mask)
                decoded_pixels = np.sum(decoded_mask)
                if decoded_pixels > 0:
                    accuracy = overlap / decoded_pixels * 100
                else:
                    accuracy = 0
            else:
                # For semantic, should match exactly
                accuracy = np.mean(decoded_mask == original_class_mask) * 100
            
            print(f"    - {cat_name}: {ann['area']:,} pixels, {accuracy:.1f}% match")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title(f"Original Image\n{img_filename}")
        axes[0].axis('off')
        
        # Original mask
        axes[1].imshow(original_mask, cmap='tab10')
        axes[1].set_title("Original Mask\n(Color = Class ID)")
        axes[1].axis('off')
        
        # Reconstructed from JSON
        reconstructed = np.zeros_like(original_mask)
        for ann in anns:
            rle = ann['segmentation']
            if isinstance(rle['counts'], str):
                rle['counts'] = rle['counts'].encode('utf-8')
            decoded_mask = mask_utils.decode(rle)
            reconstructed[decoded_mask > 0] = ann['category_id']
        
        axes[2].imshow(reconstructed, cmap='tab10')
        axes[2].set_title("Reconstructed from JSON\n(Should match left)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"verification_sample_{i+1}.png", dpi=150, bbox_inches='tight')
        print(f"  - Saved visualization: verification_sample_{i+1}.png")
    
    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Verify instance segmentation
    verify_coco_json(
        json_path="dataset/train/_annotations_instance.coco.json",
        images_dir="dataset/train/images",
        masks_dir="dataset/train/masks",
        num_samples=3
    )
    
    # Verify semantic segmentation
    verify_coco_json(
        json_path="dataset/train/_annotations_semantic.coco.json",
        images_dir="dataset/train/images",
        masks_dir="dataset/train/masks",
        num_samples=3
    )
```

---

## JSON Format Examples

### Complete Instance Segmentation JSON

**File:** `example_instance.json`

```json
{
  "info": {
    "description": "Instance Segmentation Dataset",
    "version": "1.0",
    "year": 2024
  },
  "licenses": [
    {
      "id": 1,
      "name": "Custom License",
      "url": ""
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "height": 1024,
      "width": 1024,
      "license": 1
    },
    {
      "id": 2,
      "file_name": "image_002.jpg",
      "height": 1024,
      "width": 1024,
      "license": 1
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [50, 600, 300, 200],
      "area": 58000,
      "segmentation": {
        "counts": "jQo05L3M2N2N1O1O2N1O001O0O2O1N3M4L5K7I9G",
        "size": [1024, 1024]
      },
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 1,
      "bbox": [400, 650, 250, 180],
      "area": 43000,
      "segmentation": {
        "counts": "mVo06M2N2N2O1N2O1O1O001O0O2O1N2M4L5K6J8H",
        "size": [1024, 1024]
      },
      "iscrowd": 0
    },
    {
      "id": 3,
      "image_id": 1,
      "category_id": 1,
      "bbox": [700, 620, 280, 190],
      "area": 51000,
      "segmentation": {
        "counts": "pZo04M3M2N2N1O2N1O1O001O001O0O2O1N2M4L6J8H",
        "size": [1024, 1024]
      },
      "iscrowd": 0
    },
    {
      "id": 4,
      "image_id": 1,
      "category_id": 2,
      "bbox": [0, 0, 1024, 250],
      "area": 256000,
      "segmentation": {
        "counts": "aP5o0e0bN2N2O1N2O1O1O001O001O0O2O0O2N2N3M3M3L5K6",
        "size": [1024, 1024]
      },
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "sand",
      "supercategory": "stuff",
      "isthing": 1
    },
    {
      "id": 2,
      "name": "air",
      "supercategory": "stuff",
      "isthing": 1
    },
    {
      "id": 3,
      "name": "background",
      "supercategory": "stuff",
      "isthing": 0
    }
  ]
}
```

**Key Points:**
- Image 1 has 4+ annotations (multiple per class)
- Each object is separate
- `"iscrowd": 0` for all annotations
- Can count instances per class

---

### Complete Semantic Segmentation JSON

**File:** `example_semantic.json`

```json
{
  "info": {
    "description": "Semantic Segmentation Dataset",
    "version": "1.0",
    "year": 2024
  },
  "licenses": [
    {
      "id": 1,
      "name": "Custom License",
      "url": ""
    }
  ],
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "height": 1024,
      "width": 1024,
      "license": 1
    },
    {
      "id": 2,
      "file_name": "image_002.jpg",
      "height": 1024,
      "width": 1024,
      "license": 1
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [50, 600, 930, 200],
      "area": 152000,
      "segmentation": {
        "counts": "h`o04M3M2N2N2N1O1O1O001O001O0O2O1N2M4L5K6J8H:F<D>BaA@a@O2O1O1O001O0O2N2N2N3M3M4L5K6J8H:F<D>BaA",
        "size": [1024, 1024]
      },
      "iscrowd": 1
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [0, 0, 1024, 450],
      "area": 460800,
      "segmentation": {
        "counts": "aP5o0e0bN2N2O1N2O1O1O001O001O0O2O0O2N2N3M3M3L5K6J8H:F<D>BaA@a@O2O1O1O001O0O2N2N2N3M3M4L5K",
        "size": [1024, 1024]
      },
      "iscrowd": 1
    },
    {
      "id": 3,
      "image_id": 1,
      "category_id": 3,
      "bbox": [0, 850, 1024, 174],
      "area": 178176,
      "segmentation": {
        "counts": "eX7o0c0`N2N2O1N2O1O1O001O0O2O0O2N2N3M4L5K6J8H:F<D>BaA@a@O2O",
        "size": [1024, 1024]
      },
      "iscrowd": 1
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "sand",
      "supercategory": "stuff",
      "isthing": 0
    },
    {
      "id": 2,
      "name": "air",
      "supercategory": "stuff",
      "isthing": 0
    },
    {
      "id": 3,
      "name": "background",
      "supercategory": "stuff",
      "isthing": 0
    }
  ]
}
```

**Key Points:**
- Image 1 has 3 annotations (1 per class)
- All sand objects merged into one
- `"iscrowd": 1` for all annotations
- Cannot count instances

---

## Training Configuration

### SAM3 Training Config for Instance Segmentation

**File:** `sam3_instance_train.yaml`

```yaml
# @package _global_
defaults:
  - _self_

paths:
  dataset_root: /path/to/dataset
  experiment_log_dir: /path/to/experiments/instance_seg
  bpe_path: sam3/assets/bpe_simple_vocab_16e6.txt.gz

scratch:
  enable_segmentation: True
  resolution: 1008
  train_batch_size: 2
  val_batch_size: 1
  max_data_epochs: 20

trainer:
  mode: train
  
  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          _partial_: true
        img_folder: ${paths.dataset_root}/train/images
        ann_file: ${paths.dataset_root}/train/_annotations_instance.coco.json
        load_segmentation: True
        training: true
      batch_size: ${scratch.train_batch_size}
      shuffle: True
      
    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          _partial_: true
        img_folder: ${paths.dataset_root}/valid/images
        ann_file: ${paths.dataset_root}/valid/_annotations_instance.coco.json
        load_segmentation: True
        training: false
      batch_size: ${scratch.val_batch_size}
      shuffle: False

  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${paths.bpe_path}
    eval_mode: false
    enable_segmentation: True

  loss:
    all:
      _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
      loss_fns_find:
        - _target_: sam3.train.loss.loss_fns.Boxes
          weight_dict:
            loss_bbox: 5.0
            loss_giou: 2.0
        - _target_: sam3.train.loss.loss_fns.Masks
          weight_dict:
            loss_mask: 200.0
            loss_dice: 10.0

launcher:
  num_nodes: 1
  gpus_per_node: 2
  experiment_log_dir: ${paths.experiment_log_dir}
```

---

### SAM3 Training Config for Semantic Segmentation

**File:** `sam3_semantic_train.yaml`

```yaml
# @package _global_
defaults:
  - _self_

paths:
  dataset_root: /path/to/dataset
  experiment_log_dir: /path/to/experiments/semantic_seg
  bpe_path: sam3/assets/bpe_simple_vocab_16e6.txt.gz

scratch:
  enable_segmentation: True
  resolution: 1008
  train_batch_size: 2
  val_batch_size: 1
  max_data_epochs: 20

trainer:
  mode: train
  
  data:
    train:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          _partial_: true
        img_folder: ${paths.dataset_root}/train/images
        ann_file: ${paths.dataset_root}/train/_annotations_semantic.coco.json
        load_segmentation: True
        training: true
      batch_size: ${scratch.train_batch_size}
      shuffle: True
      
    val:
      _target_: sam3.train.data.torch_dataset.TorchDataset
      dataset:
        _target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset
        coco_json_loader:
          _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
          include_negatives: true
          _partial_: true
        img_folder: ${paths.dataset_root}/valid/images
        ann_file: ${paths.dataset_root}/valid/_annotations_semantic.coco.json
        load_segmentation: True
        training: false
      batch_size: ${scratch.val_batch_size}
      shuffle: False

  model:
    _target_: sam3.model_builder.build_sam3_image_model
    bpe_path: ${paths.bpe_path}
    eval_mode: false
    enable_segmentation: True

  loss:
    all:
      _target_: sam3.train.loss.sam3_loss.Sam3LossWrapper
      loss_fns_find:
        - _target_: sam3.train.loss.loss_fns.Boxes
          weight_dict:
            loss_bbox: 5.0
            loss_giou: 2.0
        - _target_: sam3.train.loss.loss_fns.Masks
          weight_dict:
            loss_mask: 200.0
            loss_dice: 10.0
      loss_fn_semantic_seg:
        _target_: sam3.train.loss.loss_fns.SemanticSegCriterion
        weight_dict:
          loss_semantic_seg: 20.0
          loss_semantic_dice: 30.0

launcher:
  num_nodes: 1
  gpus_per_node: 2
  experiment_log_dir: ${paths.experiment_log_dir}
```

---

## Quick Start Guide

### Step 1: Organize Your Data

```bash
mkdir -p dataset/train/images dataset/train/masks
mkdir -p dataset/valid/images dataset/valid/masks
mkdir -p dataset/test/images dataset/test/masks

# Copy your images and masks to appropriate directories
```

### Step 2: Run Conversion Scripts

```bash
# For instance segmentation
python convert_to_instance_segmentation.py

# For semantic segmentation
python convert_to_semantic_segmentation.py
```

### Step 3: Verify Output

```bash
python verify_coco_json.py
```

### Step 4: Train SAM3

```bash
# Instance segmentation
python sam3/train/train.py -c sam3_instance_train.yaml

# Semantic segmentation
python sam3/train/train.py -c sam3_semantic_train.yaml
```

---

## Summary Comparison

| Aspect | Instance Segmentation | Semantic Segmentation |
|--------|----------------------|----------------------|
| **Annotations per image** | 6+ (x+y+z) | 3 (one per class) |
| **`iscrowd` value** | 0 | 1 |
| **Can count instances** | ✅ Yes | ❌ No |
| **File size** | Larger | Smaller |
| **Annotation time** | Longer | Shorter |
| **Best for** | Discrete objects | Amorphous regions |
| **Your use case** | ⚠️ Overkill | ✅ Recommended |

---

## Troubleshooting

### Issue 1: "Mask not found"

```
Solution: Ensure mask filenames match image filenames
image_001.jpg → image_001.png
```

### Issue 2: "RLE decode error"

```
Solution: Ensure masks are uint8 and use asfortranarray
binary_mask = binary_mask.astype(np.uint8)
rle = mask_utils.encode(np.asfortranarray(binary_mask))
```

### Issue 3: "Categories mismatch"

```
Solution: Check class_mapping IDs match mask pixel values
Mask pixel value 1 → category_id 1 (sand)
```

---

## Final Checklist

- [ ] Organized dataset in train/valid/test splits
- [ ] Ran conversion script (instance or semantic)
- [ ] Verified JSON with verification script
- [ ] Checked visualizations look correct
- [ ] Updated training config with correct paths
- [ ] Ready to train SAM3!

---

**End of Documentation**

This complete guide should get you from raw mask images to training-ready COCO JSON for both instance and semantic segmentation! 🎯