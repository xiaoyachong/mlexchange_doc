import os
import torch
import numpy as np
from transformers import Sam3Processor, Sam3Model, Sam3TrackerProcessor, Sam3TrackerModel

# Get device - prefer CUDA, fall back to MPS, then CPU
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
SAM3_MODEL_NAME = os.getenv("SAM3_MODEL_NAME", "facebook/sam3")


class SAM3Segmenter:
    """
    Wrapper for SAM3 segmentation with support for text, bbox, and point prompts
    """
    
    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.processor = None
        self.tracker_model = None
        self.tracker_processor = None
        
    def load_model(self, use_tracker=False):
        """Load SAM3 model and processor"""
        try:
            if use_tracker:
                if self.tracker_model is None:
                    print(f"Loading SAM3 Tracker model on {self.device}...")
                    self.tracker_model = Sam3TrackerModel.from_pretrained(SAM3_MODEL_NAME).to(self.device)
                    self.tracker_processor = Sam3TrackerProcessor.from_pretrained(SAM3_MODEL_NAME)
            else:
                if self.model is None:
                    print(f"Loading SAM3 model on {self.device}...")
                    self.model = Sam3Model.from_pretrained(SAM3_MODEL_NAME).to(self.device)
                    self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_NAME)
            return True
        except Exception as e:
            print(f"Error loading SAM3 model: {e}")
            return False
    
    def segment_with_text(self, image, text_prompt, threshold=0.5, mask_threshold=0.5):
        """
        Segment using text prompt
        Returns: list of binary masks
        """
        if not self.load_model(use_tracker=False):
            return None
            
        try:
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            return results['masks']
        except Exception as e:
            print(f"Error in text segmentation: {e}")
            return None
    
    def segment_with_boxes(self, image, boxes, threshold=0.5, mask_threshold=0.5):
        """
        Segment using bounding boxes
        boxes: list of [x1, y1, x2, y2] in pixel coordinates
        Returns: list of binary masks
        """
        if not self.load_model(use_tracker=False):
            return None
            
        try:
            # Wrap boxes in nested list for batch processing
            input_boxes = [boxes]
            input_boxes_labels = [[1] * len(boxes)]  # All positive boxes
            
            inputs = self.processor(
                images=image,
                input_boxes=input_boxes,
                input_boxes_labels=input_boxes_labels,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            return results['masks']
        except Exception as e:
            print(f"Error in box segmentation: {e}")
            return None
    
    def segment_with_points(self, image, points, labels, threshold=0.5):
        """
        Segment using point prompts
        points: list of [x, y] coordinates
        labels: list of 1 (positive) or 0 (negative) for each point
        Returns: single binary mask
        """
        if not self.load_model(use_tracker=True):
            return None
            
        try:
            inputs = self.tracker_processor(
                images=image,
                input_points=[[points]],
                input_labels=[[labels]],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.tracker_model(**inputs, multimask_output=False)
            
            masks = self.tracker_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"]
            )[0]
            
            # Return single mask (shape: [1, 1, H, W] or [1, H, W])
            mask = masks[0, 0] if len(masks.shape) == 4 else masks[0]
            return mask > threshold
        except Exception as e:
            print(f"Error in point segmentation: {e}")
            return None


def convert_sam3_mask_to_annotation(mask, color, class_id):
    """
    Convert SAM3 binary mask to annotation format compatible with the app.
    Creates a bounding box representation for UI display while storing
    the actual pixel mask for training.
    
    Returns: shape dictionary in the format expected by the app
    """
    # Find bounding box of the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Create a rectangle annotation representing the masked region
    shape = {
        "type": "rect",
        "x0": float(x_min),
        "y0": float(y_min),
        "x1": float(x_max),
        "y1": float(y_max),
        "line": {"color": color},
        "fillcolor": color,
        "editable": False,  # SAM3 masks shouldn't be edited manually
        "label": "SAM3 Auto-Annotation",
        # Store the actual pixel mask for training
        "sam3_mask": mask.astype(np.int8)
    }
    
    return shape


# Global instance
sam3_segmenter = SAM3Segmenter()
