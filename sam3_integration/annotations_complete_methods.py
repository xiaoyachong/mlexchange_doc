# ============================================================
# utils/annotations.py - REPLACE THESE TWO METHODS
# ============================================================
# Find these two methods in your existing file and replace them
# with the versions below
# ============================================================

# METHOD 1: create_annotation_mask
# Location: Around line 75-120 in the Annotations class
# ------------------------------------------------------------

def create_annotation_mask(self, sparse=False):
    """
    Create annotation mask from shapes, handling both vector graphics and SAM3 pixel masks
    """
    self.sparse = sparse
    annotation_mask = []

    image_height = self.image_shape[0]
    image_width = self.image_shape[1]

    for slice_idx, slice_data in self.annotations.items():
        slice_mask = np.full(
            [image_height, image_width], fill_value=-1, dtype=np.int8
        )
        for shape in slice_data:
            # ========== SAM3 ADDITION START ==========
            # Check if this is a SAM3 mask (has sam3_mask in the shape)
            if "sam3_mask" in shape:
                # Directly use the SAM3 pixel mask
                sam3_mask = shape["sam3_mask"]
                class_id = int(shape["class_id"])
                # Only overwrite unlabeled pixels (-1)
                slice_mask[sam3_mask > 0] = class_id
            else:
                # ========== SAM3 ADDITION END ==========
                # Handle vector graphics as before
                if shape["type"] == "Closed Freeform":
                    shape_mask = ShapeConversion.closed_path_to_array(
                        shape["svg_data"], self.image_shape, shape["class_id"]
                    )
                elif shape["type"] == "Rectangle":
                    shape_mask = ShapeConversion.rectangle_to_array(
                        shape["svg_data"], self.image_shape, shape["class_id"]
                    )
                elif shape["type"] == "Ellipse":
                    shape_mask = ShapeConversion.ellipse_to_array(
                        shape["svg_data"], self.image_shape, shape["class_id"]
                    )
                else:
                    continue
                slice_mask[shape_mask >= 0] = shape_mask[shape_mask >= 0]
        
        annotation_mask.append(slice_mask)

    if sparse:
        for idx, mask in enumerate(annotation_mask):
            annotation_mask[idx] = sp.csr_array(mask)
    self.annotation_mask = annotation_mask


# METHOD 2: _set_annotation_svg
# Location: Around line 140-155 in the Annotations class
# ------------------------------------------------------------

def _set_annotation_svg(self, annotation):
    """
    This function returns a dictionary of the svg data
    associated with a given annotation
    """
    # ========== SAM3 ADDITION START ==========
    if "sam3_mask" in annotation:
        # For SAM3 masks, we don't have SVG data
        # Store a placeholder that won't be used for mask generation
        self.svg_data = {"sam3_mask": True}
    # ========== SAM3 ADDITION END ==========
    elif "path" in annotation.keys():
        self.svg_data = {"path": annotation["path"]}
    else:
        self.svg_data = {
            "x0": annotation["x0"],
            "x1": annotation["x1"],
            "y0": annotation["y0"],
            "y1": annotation["y1"],
        }


# ============================================================
# WHAT CHANGED:
# ============================================================

# METHOD 1 (create_annotation_mask):
# - Added check for "sam3_mask" in shape
# - If SAM3 mask exists, use it directly (no conversion needed)
# - Else, use existing vector graphics conversion
# - Added 7 lines, wrapped existing code in else block

# METHOD 2 (_set_annotation_svg):
# - Added check for "sam3_mask" in annotation at the beginning
# - If SAM3 mask, store placeholder
# - Else, use existing logic
# - Added 5 lines

# ============================================================
# HOW TO APPLY:
# ============================================================
# 1. Open your existing utils/annotations.py
# 2. Find the "create_annotation_mask" method (search for "def create_annotation_mask")
# 3. Replace the entire method with METHOD 1 above
# 4. Find the "_set_annotation_svg" method (search for "def _set_annotation_svg")
# 5. Replace the entire method with METHOD 2 above
# 6. Save the file
# ============================================================
