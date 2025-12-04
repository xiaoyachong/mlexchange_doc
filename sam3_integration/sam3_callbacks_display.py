import numpy as np
from dash import Input, Output, State, callback, no_update, Patch
from PIL import Image

from constants import ANNOT_ICONS
from utils.data_utils import tiled_datasets
from utils.plot_utils import generate_notification
from utils.sam3_utils import sam3_segmenter, convert_sam3_mask_to_annotation


@callback(
    Output("sam3-prompt-input", "disabled"),
    Output("sam3-run-button", "children"),
    Input("sam3-prompt-type", "value"),
)
def update_sam3_prompt_ui(prompt_type):
    """Update UI based on selected prompt type"""
    if prompt_type == "text":
        return False, "Generate with Text"
    elif prompt_type == "bbox":
        return True, "Draw Box & Generate"
    elif prompt_type == "point":
        return True, "Click Points & Generate"
    return True, "Select Prompt Type"


@callback(
    Output("notifications-container", "children", allow_duplicate=True),
    Output({"type": "annotation-class-store", "index": 0}, "data", allow_duplicate=True),
    Output("image-viewer", "figure", allow_duplicate=True),
    Input("sam3-run-button", "n_clicks"),
    State("sam3-prompt-type", "value"),
    State("sam3-prompt-input", "value"),
    State("sam3-bbox-store", "data"),
    State("sam3-points-store", "data"),
    State("image-uri", "value"),
    State("image-selection-slider", "value"),
    State({"type": "annotation-class-store", "index": 0}, "data"),
    State("current-class-selection", "data"),
    prevent_initial_call=True,
)
def run_sam3_segmentation(
    n_clicks,
    prompt_type,
    text_prompt,
    bbox_data,
    points_data,
    image_uri,
    image_idx,
    annotation_class_store,
    current_color,
):
    """
    Main callback to run SAM3 segmentation based on user prompts
    """
    if not n_clicks:
        return no_update, no_update, no_update
    
    if not prompt_type:
        notification = generate_notification(
            "SAM3 Error",
            "red",
            ANNOT_ICONS["parameters"],
            "Please select a prompt type"
        )
        return notification, no_update, no_update
    
    # Load the current image
    try:
        image_idx_zero = image_idx - 1
        image_data = tiled_datasets.get_data_sequence_by_trimmed_uri(image_uri)[image_idx_zero]
        
        # Normalize to 0-255 range for SAM3
        low = np.percentile(image_data.ravel(), 1)
        high = np.percentile(image_data.ravel(), 99)
        image_data = np.clip((image_data - low) / (high - low), 0, 1)
        image_data = (image_data * 255).astype(np.uint8)
        
        # Convert to RGB PIL Image
        image_rgb = np.stack([image_data] * 3, axis=-1)
        pil_image = Image.fromarray(image_rgb)
        
    except Exception as e:
        notification = generate_notification(
            "SAM3 Error",
            "red",
            ANNOT_ICONS["parameters"],
            f"Error loading image: {str(e)}"
        )
        return notification, no_update, no_update
    
    # Run segmentation based on prompt type
    masks = None
    
    if prompt_type == "text":
        if not text_prompt:
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                "Please enter a text prompt"
            )
            return notification, no_update, no_update
        
        masks = sam3_segmenter.segment_with_text(pil_image, text_prompt)
        
    elif prompt_type == "bbox":
        if not bbox_data or "boxes" not in bbox_data or len(bbox_data["boxes"]) == 0:
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                "Please draw bounding boxes on the image first"
            )
            return notification, no_update, no_update
        
        boxes = bbox_data["boxes"]
        masks = sam3_segmenter.segment_with_boxes(pil_image, boxes)
        
    elif prompt_type == "point":
        if not points_data or "points" not in points_data or len(points_data["points"]) == 0:
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                "Please click points on the image first"
            )
            return notification, no_update, no_update
        
        points = points_data["points"]
        labels = points_data["labels"]
        mask = sam3_segmenter.segment_with_points(pil_image, points, labels)
        masks = [mask] if mask is not None else None
    
    # Check if segmentation was successful
    if masks is None or (isinstance(masks, list) and len(masks) == 0):
        notification = generate_notification(
            "SAM3 Error",
            "red",
            ANNOT_ICONS["parameters"],
            "SAM3 segmentation failed or found no objects"
        )
        return notification, no_update, no_update
    
    # Convert masks to annotations
    image_idx_str = str(image_idx_zero)
    if image_idx_str not in annotation_class_store["annotations"]:
        annotation_class_store["annotations"][image_idx_str] = []
    
    num_masks_added = 0
    for mask in masks:
        # Convert torch tensor to numpy if needed
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        
        # Convert mask to annotation format
        shape = convert_sam3_mask_to_annotation(
            mask, 
            current_color,
            annotation_class_store["class_id"]
        )
        
        if shape:
            annotation_class_store["annotations"][image_idx_str].append(shape)
            num_masks_added += 1
    
    # Update the figure to show new annotations
    fig = Patch()
    all_annotations = annotation_class_store["annotations"][image_idx_str]
    fig["layout"]["shapes"] = all_annotations
    
    notification = generate_notification(
        "SAM3 Success",
        "green",
        ANNOT_ICONS["results"],
        f"Added {num_masks_added} SAM3 auto-annotations"
    )
    
    return notification, annotation_class_store, fig


@callback(
    Output("sam3-bbox-store", "data"),
    Input("image-viewer", "relayoutData"),
    State("sam3-prompt-type", "value"),
    State("sam3-bbox-mode", "data"),
    State("sam3-bbox-store", "data"),
    prevent_initial_call=True,
)
def capture_sam3_bboxes(relayout_data, prompt_type, bbox_mode_active, bbox_store):
    """
    Capture bounding boxes drawn by user for SAM3
    Only active when bbox mode is enabled
    """
    if not bbox_mode_active or prompt_type != "bbox":
        return no_update
    
    # Check if a new shape was drawn
    if "shapes" in relayout_data:
        shapes = relayout_data["shapes"]
        if shapes and len(shapes) > 0:
            # Extract bbox coordinates from the last drawn shape
            last_shape = shapes[-1]
            if last_shape["type"] == "rect":
                bbox = [
                    last_shape["x0"],
                    last_shape["y0"],
                    last_shape["x1"],
                    last_shape["y1"]
                ]
                
                if bbox_store is None:
                    bbox_store = {"boxes": []}
                
                bbox_store["boxes"].append(bbox)
                return bbox_store
    
    return no_update


@callback(
    Output("sam3-points-store", "data"),
    Input("image-viewer", "clickData"),
    State("sam3-prompt-type", "value"),
    State("sam3-point-mode", "data"),
    State("sam3-points-store", "data"),
    State("sam3-point-type", "value"),
    prevent_initial_call=True,
)
def capture_sam3_points(click_data, prompt_type, point_mode_active, points_store, point_type):
    """
    Capture points clicked by user for SAM3
    Only active when point mode is enabled
    """
    if not point_mode_active or prompt_type != "point":
        return no_update
    
    if click_data and "points" in click_data:
        point = click_data["points"][0]
        x, y = point["x"], point["y"]
        
        if points_store is None:
            points_store = {"points": [], "labels": []}
        
        points_store["points"].append([x, y])
        # 1 for positive (include), 0 for negative (exclude)
        label = 1 if point_type == "positive" else 0
        points_store["labels"].append(label)
        
        return points_store
    
    return no_update


@callback(
    Output("sam3-bbox-store", "data", allow_duplicate=True),
    Output("sam3-points-store", "data", allow_duplicate=True),
    Input("sam3-clear-prompts", "n_clicks"),
    prevent_initial_call=True,
)
def clear_sam3_prompts(n_clicks):
    """Clear all SAM3 prompts"""
    if n_clicks:
        return {"boxes": []}, {"points": [], "labels": []}
    return no_update, no_update
