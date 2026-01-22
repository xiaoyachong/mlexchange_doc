import logging

import numpy as np
import torch
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ADDED: Missing function
def normalize_for_display(arr: np.ndarray, percentile_low=0.5, percentile_high=99.5) -> np.ndarray:
    """Normalize array for display with percentile-based contrast stretching."""
    arr = arr.astype(np.float32)
    
    # Use percentile-based contrast stretching
    p_low = np.percentile(arr, percentile_low)
    p_high = np.percentile(arr, percentile_high)
    
    # Clip and normalize
    arr = np.clip(arr, p_low, p_high)
    if p_high > p_low:
        arr = ((arr - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    
    return arr


def render_masks_on_patch(patch_arr: np.ndarray, masks, class_color=None):
    """
    Render masks as colored overlays on a patch image.
    
    Args:
        patch_arr: Patch image array
        masks: List of masks
        class_color: If provided, use this color for all masks (RGB tuple 0-1);
                    otherwise use rainbow colors
                    
    Returns:
        Rendered image as numpy array
    """
    patch_h, patch_w = patch_arr.shape[:2]
    result = patch_arr.copy()
    
    if masks is None or len(masks) == 0:
        return result
    
    # Create overlay
    overlay = np.zeros((patch_h, patch_w, 4), dtype=np.float32)
    
    num_masks = len(masks)
    for i, m in enumerate(masks):
        # Handle mask dimensions
        if m.ndim == 3:
            m = m.squeeze(0)
        
        # Resize if needed
        if m.shape != (patch_h, patch_w):
            m_tensor = torch.from_numpy(m).float()[None, None, ...]
            m_tensor = torch.nn.functional.interpolate(
                m_tensor, size=(patch_h, patch_w), mode="bilinear", align_corners=False
            )
            m = m_tensor.squeeze().numpy()
        
        # Generate color
        if class_color is not None:
            color = class_color
        else:
            color = plt.cm.rainbow(i / max(num_masks, 1))
        
        # Add to overlay
        mask_binary = m > 0.5
        overlay[mask_binary, 0] = color[0]
        overlay[mask_binary, 1] = color[1]
        overlay[mask_binary, 2] = color[2]
        overlay[mask_binary, 3] = 0.45
    
    # Blend overlay with original image
    alpha = overlay[:, :, 3:4]
    result = result * (1 - alpha) + overlay[:, :, :3] * 255 * alpha
    result = result.astype(np.uint8)
    
    return result


def stitch_patch_results(patch_results, orig_h, orig_w, n_rows, n_cols, patch_size=512):
    """
    Stitch rendered patch results back into a full image.
    
    Args:
        patch_results: List of (rendered_patch, row, col, y_start, x_start, patch_h, patch_w)
        orig_h: Original image height
        orig_w: Original image width
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        patch_size: Size of patches
        
    Returns:
        Stitched full-size image
    """
    stitched = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    
    for rendered_patch, row, col, y_start, x_start, patch_h, patch_w in patch_results:
        # Crop to actual size (remove padding)
        rendered_crop = rendered_patch[:patch_h, :patch_w]
        
        # Place in stitched image
        stitched[y_start:y_start+patch_h, x_start:x_start+patch_w] = rendered_crop
    
    return stitched


def create_class_level_visualization(
    patch_results_dict,
    orig_h,
    orig_w, 
    image_arr,
    class_colors,
    patch_size=512
    ):
    """
    Create a single class-level visualization with all classes overlaid.
    
    Args:
        patch_results_dict: Dict mapping prompt to list of 
                           (masks, scores, row, col, y, x, h, w) tuples
        orig_h: Original image height
        orig_w: Original image width
        image_arr: Original image array
        class_colors: Dict mapping prompt to RGB color tuple (0-1)
        patch_size: Size of patches
        
    Returns:
        Class-level visualization image
    """
    # Start with the original image
    result = normalize_for_display(image_arr)
    
    # Create a combined overlay for all classes
    overlay = np.zeros((orig_h, orig_w, 4), dtype=np.float32)
    
    for prompt, patch_results in patch_results_dict.items():
        color = class_colors[prompt]
        logger.info(f"    Adding class '{prompt}' in color {color}")
        
        for masks, scores, row, col, y_start, x_start, patch_h, patch_w in patch_results:
            if masks is None or len(masks) == 0:
                continue
            
            for m in masks:
                # Handle mask dimensions
                if m.ndim == 3:
                    m = m.squeeze(0)
                
                # Resize to patch size if needed
                if m.shape != (patch_size, patch_size):
                    m_tensor = torch.from_numpy(m).float()[None, None, ...]
                    m_tensor = torch.nn.functional.interpolate(
                        m_tensor, size=(patch_size, patch_size), mode="bilinear", align_corners=False
                    )
                    m = m_tensor.squeeze().numpy()
                
                # Crop to actual patch size
                m_crop = m[:patch_h, :patch_w]
                
                # Place in full overlay
                mask_binary = m_crop > 0.5
                y_end = y_start + patch_h
                x_end = x_start + patch_w
                
                overlay[y_start:y_end, x_start:x_end][mask_binary, 0] = color[0]
                overlay[y_start:y_end, x_start:x_end][mask_binary, 1] = color[1]
                overlay[y_start:y_end, x_start:x_end][mask_binary, 2] = color[2]
                overlay[y_start:y_end, x_start:x_end][mask_binary, 3] = 0.45
    
    # Blend overlay with original image
    alpha = overlay[:, :, 3:4]
    result = result[:, :, None] if result.ndim == 2 else result
    if result.shape[2] == 1:
        result = np.repeat(result, 3, axis=2)
    
    result = result * (1 - alpha) + overlay[:, :, :3] * 255 * alpha
    result = result.astype(np.uint8)
    
    return result


def render_all_objects(image_arr, all_masks_list, orig_h, orig_w, patch_size=512):
    """
    Render all object masks with rainbow colors.
    
    Args:
        image_arr: Original image array
        all_masks_list: List of (mask, y_start, x_start, patch_h, patch_w) tuples
        orig_h: Original image height
        orig_w: Original image width
        patch_size: Size of patches
        
    Returns:
        Rendered image with all objects
    """
    result = normalize_for_display(image_arr)
    if result.ndim == 2:
        result = np.stack([result] * 3, axis=-1)
    
    overlay = np.zeros((orig_h, orig_w, 4), dtype=np.float32)
    num_total_masks = len(all_masks_list)
    
    for idx, (m, y_start, x_start, patch_h, patch_w) in enumerate(all_masks_list):
        # Generate rainbow color for this mask
        color = plt.cm.rainbow(idx / max(num_total_masks, 1))
        
        # Handle mask dimensions
        if m.ndim == 3:
            m = m.squeeze(0)
        
        # Resize to patch size if needed
        if m.shape != (patch_size, patch_size):
            m_tensor = torch.from_numpy(m).float()[None, None, ...]
            m_tensor = torch.nn.functional.interpolate(
                m_tensor, size=(patch_size, patch_size), mode="bilinear", align_corners=False
            )
            m = m_tensor.squeeze().numpy()
        
        # Crop to actual patch size
        m_crop = m[:patch_h, :patch_w]
        
        # Place in full overlay
        mask_binary = m_crop > 0.5
        y_end = y_start + patch_h
        x_end = x_start + patch_w
        
        overlay[y_start:y_end, x_start:x_end][mask_binary, 0] = color[0]
        overlay[y_start:y_end, x_start:x_end][mask_binary, 1] = color[1]
        overlay[y_start:y_end, x_start:x_end][mask_binary, 2] = color[2]
        overlay[y_start:y_end, x_start:x_end][mask_binary, 3] = 0.45
    
    # Blend overlay with original image
    alpha = overlay[:, :, 3:4]
    result = result * (1 - alpha) + overlay[:, :, :3] * 255 * alpha
    result = result.astype(np.uint8)
    
    return result


def plot_combined_results(
    object_results,
    object_combined,
    class_results,
    class_level_img, 
    model_name,
    class_colors
    ):
    """
    Plot both object-level and class-level visualizations.
    
    Args:
        object_results: Dict of prompt -> stitched image (object-level, per prompt)
        object_combined: Combined object-level image (all prompts)
        class_results: Dict of prompt -> stitched image (class-level, per prompt)
        class_level_img: Class-level visualization (all classes combined)
        model_name: Name for the figure title
        class_colors: Dict mapping prompt to RGB color tuple (0-1)
    """
    num_prompts = len(object_results)
    ncols = 3
    nrows_per_section = (num_prompts + ncols - 1) // ncols
    
    # Total rows: object-level rows + class-level rows
    total_rows = nrows_per_section + nrows_per_section
    
    fig = plt.figure(figsize=(18, 6 * total_rows))
    gs = fig.add_gridspec(total_rows, ncols, hspace=0.3, wspace=0.2)
    
    current_row = 0
    
    # === OBJECT-LEVEL RESULTS (per prompt) ===
    for idx, (prompt, stitched_img) in enumerate(object_results.items()):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[current_row + row, col])
        
        ax.imshow(stitched_img)
        ax.set_title(f"Object-Level: '{prompt}'", fontsize=12, fontweight="bold")
        ax.axis("off")
    
    # === OBJECT-LEVEL COMBINED (at last position) ===
    # Place at row 1, col 2 (last column of second row)
    ax_object_combined = fig.add_subplot(gs[current_row + 1, 2])
    ax_object_combined.imshow(object_combined)
    ax_object_combined.set_title(f"Object-Level:\nAll Objects Combined", 
                                 fontsize=12, fontweight="bold")
    ax_object_combined.axis("off")
    
    current_row += nrows_per_section
    
    # === CLASS-LEVEL RESULTS (per prompt) ===
    for idx, (prompt, stitched_img) in enumerate(class_results.items()):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[current_row + row, col])
        
        ax.imshow(stitched_img)
        color = class_colors[prompt]
        ax.set_title(f"Class-Level: '{prompt}'", fontsize=12, fontweight="bold")
        # Add color indicator with smaller font
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.05, 0.05, 
                                   transform=ax.transAxes,
                                   facecolor=color, edgecolor='white', linewidth=2))
        ax.axis("off")
    
    # === CLASS-LEVEL COMBINED (at last position) ===
    # Place at row (current_row + 1), col 2 (last column of second row)
    ax_class = fig.add_subplot(gs[current_row + 1, 2])
    ax_class.imshow(class_level_img)
    
    # Create legend for class-level with smaller font
    legend_elements = []
    for prompt, color in class_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                         markerfacecolor=color, markersize=5, label=prompt))
    ax_class.legend(handles=legend_elements, loc='upper right', fontsize=5)
    ax_class.set_title(f"Class-Level:\nAll Classes Combined", 
                       fontsize=12, fontweight="bold")
    ax_class.axis("off")
    
    fig.suptitle(model_name, fontsize=16, fontweight="bold", y=0.995)
    plt.show()


def create_combined_visualizations(
    raw_patch_results,
    image_arr,
    class_colors, 
    orig_h,
    orig_w,
    n_rows,
    n_cols,
    patch_size=512
    ):
    """
    Create combined visualizations for both object-level and class-level.
    
    Args:
        raw_patch_results: Dict of prompt -> raw results list
        image_arr: Original image array
        class_colors: Dict mapping prompts to RGB tuples
        orig_h: Original image height
        orig_w: Original image width
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        patch_size: Size of patches
        
    Returns:
        Tuple of (object_combined, class_combined) images
    """
    
    # Create class-level combined
    logger.info("\n  Creating class-level visualization (all classes combined)...")
    class_level_combined = create_class_level_visualization(
        raw_patch_results, orig_h, orig_w, n_rows, n_cols,
        image_arr, class_colors, patch_size
    )
    
    # Create object-level combined
    logger.info("\n  Creating object-level visualization (all objects combined)...")
    all_masks = []
    
    for prompt, raw_results in raw_patch_results.items():
        for masks, scores, row, col, y_start, x_start, patch_h, patch_w in raw_results:
            if masks is not None and len(masks) > 0:
                for m in masks:
                    all_masks.append((m, y_start, x_start, patch_h, patch_w))
    
    object_combined = render_all_objects(image_arr, all_masks, orig_h, orig_w, patch_size)
    
    return object_combined, class_level_combined
