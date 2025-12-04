# ============================================================
# components/control_bar.py - CHANGES NEEDED
# ============================================================

# CHANGE 1: Add import (around line 10-20)
# ------------------------------------------------------------
# BEFORE:
"""
from components.annotation_class import annotation_class_item
from components.parameter_items import ControlItem
from constants import ANNOT_ICONS, KEYBINDS
from utils.data_utils import models, tiled_datasets
"""

# AFTER (add the highlighted line):
"""
from components.annotation_class import annotation_class_item
from components.parameter_items import ControlItem
from components.sam3_section import sam3_annotation_section  # <- ADD THIS
from constants import ANNOT_ICONS, KEYBINDS
from utils.data_utils import models, tiled_datasets
"""


# CHANGE 2: Add SAM3 accordion item (around line 100-300)
# ------------------------------------------------------------
# BEFORE:
"""
                    children=[
                        _accordion_item(
                            "Data selection",
                            ...
                        ),
                        _accordion_item(
                            "Image transformations",
                            ...
                        ),
                        _accordion_item(
                            "Annotation tools",
                            "mdi:paintbrush-outline",
                            "annotations",
                            id="annotations-controls",
                            children=[...],
                        ),
                        _accordion_item(
                            "Model configuration",
                            "carbon:ibm-watson-machine-learning",
                            "run-model",
                            id="model-configuration",
                            children=[...],
                        ),
                    ],
"""

# AFTER (insert the SAM3 item between Annotation tools and Model configuration):
"""
                    children=[
                        _accordion_item(
                            "Data selection",
                            ...
                        ),
                        _accordion_item(
                            "Image transformations",
                            ...
                        ),
                        _accordion_item(
                            "Annotation tools",
                            "mdi:paintbrush-outline",
                            "annotations",
                            id="annotations-controls",
                            children=[...],
                        ),
                        # ========== ADD THIS BLOCK ==========
                        _accordion_item(
                            "SAM3 Auto-Annotation",
                            "mdi:robot",
                            "sam3-auto-annotation",
                            id="sam3-annotation-controls",
                            children=[
                                sam3_annotation_section()
                            ],
                        ),
                        # ====================================
                        _accordion_item(
                            "Model configuration",
                            "carbon:ibm-watson-machine-learning",
                            "run-model",
                            id="model-configuration",
                            children=[...],
                        ),
                    ],
"""

# ============================================================
# SUMMARY: 
# - Add 1 import line
# - Add 1 accordion item (8 lines) in the children list
# Total: 9 lines added
# ============================================================
