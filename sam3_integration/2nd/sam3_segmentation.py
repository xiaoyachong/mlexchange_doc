import logging
import random
import numpy as np
import plotly.express as px
from dash import Input, Output, State, callback, no_update, Patch, callback_context, ALL, MATCH, html
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
from PIL import Image

from components.sam3_class_item import sam3_class_item
from constants import ANNOT_ICONS
from utils.data_utils import tiled_datasets
from utils.plot_utils import generate_notification, generate_notification_bg_icon_col
from utils.sam3_utils import sam3_segmenter, convert_sam3_mask_to_annotation

# Set up logging
logger = logging.getLogger(__name__)


# ========== CLASS MANAGEMENT CALLBACKS ==========
@callback(
    Output({"type": "annotation-class", "index": ALL}, "style", allow_duplicate=True),
    Output({"type": "sam3-class", "index": ALL}, "style", allow_duplicate=True),
    Output("closed-freeform", "disabled"),
    Output("circle", "disabled"),
    Output("rectangle", "disabled"),
    Output("pan-and-zoom", "disabled"),
    Output("clear-all", "disabled"),
    Output("generate-annotation-class", "disabled"),
    Output("sam3-pan-zoom", "disabled"),
    Output("sam3-text-mode", "disabled"),
    Output("sam3-bbox-mode", "disabled"),
    Output("sam3-point-mode", "disabled"),
    Output("sam3-generate-button", "disabled"),
    Output("sam3-clear-button", "disabled"),
    Output("sam3-generate-class", "disabled"),
    Input({"type": "annotation-class", "index": ALL}, "n_clicks"),
    Input({"type": "sam3-class", "index": ALL}, "n_clicks"),
    State({"type": "annotation-class-store", "index": ALL}, "data"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    State("current-class-selection", "data"),
    State("sam3-current-class-selection", "data"),
    prevent_initial_call=True,
)
def manage_annotation_mode_exclusion(
    annot_clicks,
    sam3_clicks,
    all_annotation_classes,
    all_sam3_classes,
    current_annot_color,
    current_sam3_color,
):
    """
    Ensure manual annotation and SAM3 auto-annotation are mutually exclusive.
    When one is active, the other is disabled.
    """
    trigger = callback_context.triggered_id
    
    default_annot_style = {
        "border": "1px solid #EAECEF",
        "borderRadius": "3px",
        "marginBottom": "4px",
        "display": "flex",
        "justifyContent": "space-between",
    }
    selected_annot_style = {**default_annot_style, "backgroundColor": "#EAECEF"}
    disabled_annot_style = {**default_annot_style, "opacity": "0.5", "pointerEvents": "none"}
    
    default_sam3_style = {
        "border": "1px solid #EAECEF",
        "borderRadius": "3px",
        "marginBottom": "4px",
        "display": "flex",
        "justifyContent": "space-between",
    }
    selected_sam3_style = {**default_sam3_style, "backgroundColor": "#EAECEF"}
    disabled_sam3_style = {**default_sam3_style, "opacity": "0.5", "pointerEvents": "none"}
    
    # Determine which system was clicked
    if trigger and trigger.get("type") == "annotation-class":
        # Manual annotation class was clicked - enable manual, disable SAM3
        logger.info("Manual annotation mode activated")
        
        # Style annotation classes
        annot_styles = []
        for a_class in all_annotation_classes:
            if a_class["color"] == current_annot_color:
                annot_styles.append(selected_annot_style)
            else:
                annot_styles.append(default_annot_style)
        
        # Disable all SAM3 classes
        sam3_styles = [disabled_sam3_style] * len(all_sam3_classes)
        
        return (
            annot_styles,
            sam3_styles,
            False, False, False, False, False, False,  # Enable manual tools
            True, True, True, True, True, True, True,   # Disable SAM3 tools
        )
    
    elif trigger and trigger.get("type") == "sam3-class":
        # SAM3 class was clicked - enable SAM3, disable manual
        logger.info("SAM3 auto-annotation mode activated")
        
        # Disable all annotation classes
        annot_styles = [disabled_annot_style] * len(all_annotation_classes)
        
        # Style SAM3 classes
        sam3_styles = []
        for s_class in all_sam3_classes:
            if s_class["color"] == current_sam3_color:
                sam3_styles.append(selected_sam3_style)
            else:
                sam3_styles.append(default_sam3_style)
        
        return (
            annot_styles,
            sam3_styles,
            True, True, True, True, True, True,        # Disable manual tools
            False, False, False, False, False, False, False,  # Enable SAM3 tools
        )
    
    # Fallback - should not reach here
    return (
        [default_annot_style] * len(all_annotation_classes),
        [default_sam3_style] * len(all_sam3_classes),
        False, False, False, False, False, False,
        False, False, False, False, False, False, False,
    )

@callback(
    Output("sam3-generate-class-modal", "opened"),
    Output("sam3-create-class", "disabled"),
    Output("sam3-bad-label-color", "children"),
    Input("sam3-generate-class", "n_clicks"),
    Input("sam3-create-class", "n_clicks"),
    Input("sam3-class-label", "value"),
    State("sam3-generate-class-modal", "opened"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    prevent_initial_call=True,
)
def open_sam3_class_modal(generate, create, new_label, opened, all_sam3_classes):
    """Open/close modal for creating new SAM3 class"""
    current_classes = [a["label"] for a in all_sam3_classes] if all_sam3_classes else []
    
    if callback_context.triggered[0]["prop_id"] == "sam3-class-label.value":
        disable_class_creation = False
        error_msg = []
        if new_label in current_classes:
            disable_class_creation = True
            error_msg.append("Label Already in Use!")
            error_msg.append(html.Br())
        if new_label == "" or new_label is None:
            disable_class_creation = True
        if new_label == "Unlabeled":
            disable_class_creation = True
            error_msg.append("Label name cannot be 'Unlabeled'")
        return no_update, disable_class_creation, error_msg
    
    return not opened, False, ""


@callback(
    Output("sam3-class-label", "value"),
    Output("sam3-class-container", "children", allow_duplicate=True),
    Output("sam3-current-class-selection", "data", allow_duplicate=True),
    Output("notifications-container", "children", allow_duplicate=True),
    Input("sam3-create-class", "n_clicks"),
    State("sam3-class-container", "children"),
    State("sam3-class-label", "value"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    prevent_initial_call=True,
)
def add_sam3_class(create, current_classes, new_class_label, all_sam3_classes):
    """Add new SAM3 class"""
    # Generate random color
    current_colors = [a["color"] for a in all_sam3_classes] if all_sam3_classes else []
    color_suggestions = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet
    available_colors = [c for c in color_suggestions if c not in current_colors]
    new_class_color = random.choice(available_colors) if available_colors else "#DB0606"
    
    existing_ids = [annotation["class_id"] for annotation in all_sam3_classes] if all_sam3_classes else []
    
    # Create new SAM3 class item
    new_class = sam3_class_item(new_class_color, new_class_label, existing_ids)
    current_classes.append(new_class)
    
    logger.info(f"Created new SAM3 class: {new_class_label} with color {new_class_color}")
    
    notification = generate_notification_bg_icon_col(
        f"{new_class_label} class created & selected", new_class_color, "mdi:color"
    )
    
    return "", current_classes, new_class_color, notification


@callback(
    Output("sam3-current-class-selection", "data", allow_duplicate=True),
    Output("notifications-container", "children", allow_duplicate=True),
    Input({"type": "sam3-class", "index": ALL}, "n_clicks"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    State("sam3-current-class-selection", "data"),
    prevent_initial_call=True,
)
def update_sam3_current_class_selection(class_selected, all_sam3_classes, previous_selection):
    """Update current SAM3 class selection when a class is clicked"""
    
    logger.info(f"SAM3 class click detected!")
    logger.info(f"Triggered ID: {callback_context.triggered_id}")
    logger.info(f"All SAM3 classes: {[(c['label'], c['color']) for c in all_sam3_classes]}")
    logger.info(f"Previous selection: {previous_selection}")
    
    current_selection = None
    label_name = None
    
    if callback_context.triggered_id:
        if len(callback_context.triggered) == 1:
            for c in all_sam3_classes:
                if c["class_id"] == callback_context.triggered_id["index"]:
                    current_selection = c["color"]
                    label_name = c["label"]
                    logger.info(f"Selected class: {label_name} with color {current_selection}")
        elif len(all_sam3_classes) > 0:
            current_selection = all_sam3_classes[-1]["color"]
            label_name = all_sam3_classes[-1]["label"]
            logger.info(f"Auto-selected last class: {label_name} with color {current_selection}")
    
    if previous_selection == current_selection:
        logger.info("Selection unchanged, preventing update")
        raise PreventUpdate
    
    logger.info(f"Updating selection to: {current_selection}")
    
    notification = generate_notification_bg_icon_col(
        f"{label_name} class selected", current_selection, "mdi:color"
    )
    
    return current_selection, notification


@callback(
    Output({"type": "sam3-class", "index": ALL}, "style"),
    Input("sam3-current-class-selection", "data"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
)
def update_sam3_selected_class_style(selected_class, all_sam3_classes):
    """Update style of selected SAM3 class"""
    default_style = {
        "border": "1px solid #EAECEF",
        "borderRadius": "3px",
        "marginBottom": "4px",
        "display": "flex",
        "justifyContent": "space-between",
    }
    selected_style = {
        "border": "1px solid #EAECEF",
        "borderRadius": "3px",
        "marginBottom": "4px",
        "display": "flex",
        "justifyContent": "space-between",
        "backgroundColor": "#EAECEF",
    }
    
    ids = [c["color"] for c in all_sam3_classes]
    if selected_class in ids:
        index = ids.index(selected_class)
        styles = [default_style] * len(ids)
        styles[index] = selected_style
        return styles
    else:
        styles = [default_style] * len(ids)
        if len(styles) > 0:
            styles[-1] = selected_style
        return styles


@callback(
    Output("image-viewer", "figure", allow_duplicate=True),
    Input({"type": "sam3-edit-class-store", "index": ALL}, "data"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    State("image-selection-slider", "value"),
    prevent_initial_call=True,
)
def re_draw_sam3_annotations_after_editing_class_color(
    hide_show_click, all_sam3_class_store, image_idx
):
    """Re-draw SAM3 annotations after editing class color"""
    fig = Patch()
    image_idx = str(image_idx - 1)
    all_annotations = []
    for a in all_sam3_class_store:
        if a["is_visible"] and "annotations" in a and image_idx in a["annotations"]:
            all_annotations += a["annotations"][image_idx]
    fig["layout"]["shapes"] = all_annotations
    return fig


@callback(
    Output({"type": "sam3-class-label", "index": MATCH}, "children"),
    Output({"type": "sam3-class-color", "index": MATCH}, "style"),
    Output({"type": "sam3-class-store", "index": MATCH}, "data"),
    Output({"type": "sam3-class", "index": MATCH}, "n_clicks"),
    Output({"type": "sam3-edit-class-store", "index": MATCH}, "data"),
    Input({"type": "sam3-save-edited-class-btn", "index": MATCH}, "n_clicks"),
    State({"type": "sam3-edit-class-text-input", "index": MATCH}, "value"),
    State({"type": "sam3-edit-class-colorpicker", "index": MATCH}, "value"),
    State({"type": "sam3-class-store", "index": MATCH}, "data"),
    prevent_initial_call=True,
)
def edit_sam3_class(edit_clicked, new_label, new_color, sam3_class_store):
    """Edit SAM3 class name and color"""
    sam3_class_store["label"] = new_label
    sam3_class_store["color"] = new_color
    class_color_identifier = {
        "width": "25px",
        "height": "25px",
        "backgroundColor": new_color + "50",
        "margin": "5px",
        "borderRadius": "3px",
        "border": f"2px solid {new_color}",
    }
    # Update color in annotations
    for img_idx, annots in sam3_class_store["annotations"].items():
        for annots in sam3_class_store["annotations"][img_idx]:
            annots["line"]["color"] = new_color
            if "fillcolor" in annots:
                annots["fillcolor"] = new_color

    return new_label, class_color_identifier, sam3_class_store, 1, True


@callback(
    Output({"type": "sam3-edit-class-modal", "index": MATCH}, "opened"),
    Output({"type": "sam3-save-edited-class-btn", "index": MATCH}, "disabled"),
    Output({"type": "sam3-bad-edit-label", "index": MATCH}, "children"),
    Output({"type": "sam3-edit-class-modal", "index": MATCH}, "title"),
    Output(
        {"type": "sam3-edit-class-text-input", "index": MATCH},
        "value",
        allow_duplicate=True,
    ),
    Output(
        {"type": "sam3-edit-class-colorpicker", "index": MATCH},
        "value",
        allow_duplicate=True,
    ),
    Input({"type": "sam3-edit-class", "index": MATCH}, "n_clicks"),
    Input({"type": "sam3-save-edited-class-btn", "index": MATCH}, "n_clicks"),
    Input({"type": "sam3-edit-class-text-input", "index": MATCH}, "value"),
    Input({"type": "sam3-edit-class-colorpicker", "index": MATCH}, "value"),
    State({"type": "sam3-edit-class-modal", "index": MATCH}, "opened"),
    State({"type": "sam3-class-store", "index": MATCH}, "data"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    prevent_initial_call=True,
)
def open_edit_sam3_class_modal(
    edit_button,
    edit_modal,
    new_label,
    new_color,
    opened,
    class_to_edit,
    all_sam3_class_store,
):
    """Open/close modal for editing SAM3 class"""
    modal_title = f"Edit class: {class_to_edit['label']}"
    
    if callback_context.triggered_id["type"] == "sam3-edit-class":
        return (
            not opened,
            no_update,
            no_update,
            no_update,
            class_to_edit["label"],
            class_to_edit["color"],
        )
    
    if callback_context.triggered_id["type"] in [
        "sam3-edit-class-text-input",
        "sam3-edit-class-colorpicker",
    ]:
        current_classes = [a["label"] for a in all_sam3_class_store]
        current_colors = [a["color"] for a in all_sam3_class_store]
        current_classes.remove(class_to_edit["label"])
        current_colors.remove(class_to_edit["color"])
        edit_disabled = False
        error_msg = []
        if new_label in current_classes:
            error_msg.append("Label Already in Use!")
            error_msg.append(html.Br())
            edit_disabled = True
        if new_label == "":
            error_msg.append("Label name cannot be empty!")
            error_msg.append(html.Br())
            edit_disabled = True
        if new_label == "Unlabeled":
            error_msg.append("Label name cannot be 'Unlabeled'!")
            error_msg.append(html.Br())
            edit_disabled = True
        if new_color in current_colors:
            error_msg.append("Color Already in use!")
            edit_disabled = True
        return no_update, edit_disabled, error_msg, modal_title, no_update, no_update
    
    return not opened, False, no_update, modal_title, no_update, no_update


@callback(
    Output({"type": "sam3-delete-class-modal", "index": MATCH}, "opened"),
    Output({"type": "sam3-delete-class-modal", "index": MATCH}, "title"),
    Output({"type": "sam3-cannot-delete-last-class-modal", "index": MATCH}, "opened"),
    Input({"type": "sam3-delete-class", "index": MATCH}, "n_clicks"),
    Input({"type": "sam3-confirm-delete-class-btn", "index": MATCH}, "n_clicks"),
    Input({"type": "sam3-cancel-delete-class-btn", "index": MATCH}, "n_clicks"),
    Input({"type": "sam3-ok-not-delete-last-class-btn", "index": MATCH}, "n_clicks"),
    State({"type": "sam3-cannot-delete-last-class-modal", "index": MATCH}, "opened"),
    State({"type": "sam3-delete-class-modal", "index": MATCH}, "opened"),
    State({"type": "sam3-class-store", "index": MATCH}, "data"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    prevent_initial_call=True,
)
def open_delete_sam3_class_modal(
    remove_class,
    continue_remove_class_modal,
    cancel_remove_class_modal,
    ok_not_delete_modal,
    cannot_delete_modal_opened,
    delete_modal_opened,
    class_to_delete,
    all_sam3_classes,
):
    """Open/close modal for deleting SAM3 class"""
    if len(all_sam3_classes) == 1:
        return no_update, no_update, not cannot_delete_modal_opened
    modal_title = f"Delete class: {class_to_delete['label']}"
    return not delete_modal_opened, modal_title, no_update


@callback(
    Output("sam3-class-container", "children"),
    Input({"type": "sam3-deleted-class-store", "index": ALL}, "data"),
    State("sam3-class-container", "children"),
    prevent_initial_call=True,
)
def delete_sam3_class(is_deleted, all_classes):
    """Delete SAM3 class from container"""
    is_deleted = [x for x in is_deleted if x is not None]
    if is_deleted:
        is_deleted = is_deleted[0]
        updated_classes = [
            c for c in all_classes if c["props"]["id"]["index"] != is_deleted
        ]
        return updated_classes
    return no_update


@callback(
    Output({"type": "sam3-deleted-class-store", "index": MATCH}, "data"),
    Input({"type": "sam3-confirm-delete-class-btn", "index": MATCH}, "n_clicks"),
    State({"type": "sam3-class-store", "index": MATCH}, "data"),
    prevent_initial_call=True,
)
def clear_sam3_class(remove, sam3_class_store):
    """Mark SAM3 class for deletion"""
    deleted_class = sam3_class_store["class_id"]
    return deleted_class


@callback(
    Output("image-viewer", "figure", allow_duplicate=True),
    Input({"type": "sam3-hide-show-class-store", "index": ALL}, "data"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    State("image-selection-slider", "value"),
    prevent_initial_call=True,
)
def hide_show_sam3_annotations_on_fig(
    hide_show_click, all_sam3_class_store, image_idx
):
    """Hide or show SAM3 annotations"""
    fig = Patch()
    image_idx = str(image_idx - 1)
    all_annotations = []
    for a in all_sam3_class_store:
        if a["is_visible"] and "annotations" in a and image_idx in a["annotations"]:
            all_annotations += a["annotations"][image_idx]
    fig["layout"]["shapes"] = all_annotations
    return fig


@callback(
    Output(
        {"type": "sam3-class-store", "index": MATCH}, "data", allow_duplicate=True
    ),
    Output({"type": "sam3-hide-show-class-store", "index": MATCH}, "data"),
    Output({"type": "sam3-hide-class", "index": MATCH}, "children"),
    Input({"type": "sam3-hide-class", "index": MATCH}, "n_clicks"),
    State({"type": "sam3-class-store", "index": MATCH}, "data"),
    State({"type": "sam3-hide-show-class-store", "index": MATCH}, "data"),
    prevent_initial_call=True,
)
def hide_show_sam3_class(
    hide_show_click,
    sam3_class_store,
    hide_show_class_store,
):
    """Toggle visibility of SAM3 class"""
    is_visible = sam3_class_store["is_visible"]
    sam3_class_store["is_visible"] = not is_visible
    hide_show_class_store["is_visible"] = not is_visible
    if is_visible:
        updated_icon = DashIconify(icon="mdi:hide")
    else:
        updated_icon = DashIconify(icon="mdi:eye")
    return sam3_class_store, hide_show_class_store, updated_icon


# ========== MODE SELECTION CALLBACK ==========

@callback(
    Output("sam3-pan-zoom", "style"),
    Output("sam3-text-mode", "style"),
    Output("sam3-bbox-mode", "style"),
    Output("sam3-point-mode", "style"),
    Output("sam3-text-input-container", "style"),
    Output("sam3-active-mode", "data"),
    Output("image-viewer", "figure", allow_duplicate=True),
    Input("sam3-pan-zoom", "n_clicks"),
    Input("sam3-text-mode", "n_clicks"),
    Input("sam3-bbox-mode", "n_clicks"),
    Input("sam3-point-mode", "n_clicks"),
    State("sam3-current-class-selection", "data"),
    prevent_initial_call=True,
)
def sam3_mode_selection(pan_clicks, text_clicks, bbox_clicks, point_clicks, sam3_color):
    """Handle SAM3 mode selection"""
    
    trigger = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
    
    active = {"backgroundColor": "#EAECEF"}
    inactive = {"border": "1px solid white"}
    
    styles = {
        "sam3-pan-zoom": inactive,
        "sam3-text-mode": inactive,
        "sam3-bbox-mode": inactive,
        "sam3-point-mode": inactive,
    }
    
    patched_figure = Patch()
    mode = None
    text_container_style = {"display": "none"}
    
    if trigger == "sam3-pan-zoom" and pan_clicks > 0:
        patched_figure["layout"]["dragmode"] = "pan"
        styles[trigger] = active
        mode = None
    elif trigger == "sam3-text-mode" and text_clicks > 0:
        styles[trigger] = active
        mode = "text"
        text_container_style = {"display": "block"}
    elif trigger == "sam3-bbox-mode" and bbox_clicks > 0:
        patched_figure["layout"]["dragmode"] = "drawrect"
        # Set SAM3 color for drawing
        patched_figure["layout"]["newshape"]["line"]["color"] = sam3_color
        patched_figure["layout"]["newshape"]["fillcolor"] = sam3_color
        patched_figure["layout"]["newshape"]["editable"] = False
        patched_figure["layout"]["newshape"]["label"] = "SAM3_bbox"
        styles[trigger] = active
        mode = "bbox"
    elif trigger == "sam3-point-mode" and point_clicks > 0:
        patched_figure["layout"]["dragmode"] = "pan"
        styles[trigger] = active
        mode = "point"
    
    return (
        styles["sam3-pan-zoom"],
        styles["sam3-text-mode"],
        styles["sam3-bbox-mode"],
        styles["sam3-point-mode"],
        text_container_style,
        mode,
        patched_figure,
    )


# ========== SEGMENTATION CALLBACK ==========

@callback(
    Output("notifications-container", "children", allow_duplicate=True),
    Output({"type": "sam3-class-store", "index": ALL}, "data", allow_duplicate=True),
    Output("image-viewer", "figure", allow_duplicate=True),
    Output("sam3-bbox-store", "data", allow_duplicate=True),
    Output("sam3-points-store", "data", allow_duplicate=True),
    Input("sam3-generate-button", "n_clicks"),
    State("sam3-active-mode", "data"),
    State("sam3-text-prompt", "value"),
    State("sam3-points-store", "data"),
    State("image-uri", "value"),
    State("image-selection-slider", "value"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    State("sam3-current-class-selection", "data"),
    State("image-viewer", "figure"),
    prevent_initial_call=True,
)
def run_sam3_segmentation(
    n_clicks,
    active_mode,
    text_prompt,
    points_data,
    image_uri,
    image_idx,
    all_sam3_classes,
    sam3_current_color,
    fig,
):
    """Main callback to run SAM3 segmentation"""
    
    if not n_clicks:
        return no_update, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
    
    logger.info(f"SAM3 Generate clicked - Mode: {active_mode}")
    logger.info(f"SAM3 current color: {sam3_current_color}")
    
    if not active_mode:
        notification = generate_notification(
            "SAM3 Error",
            "red",
            ANNOT_ICONS["parameters"],
            "Please select a mode first"
        )
        return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
    
    # Find current selected SAM3 class
    selected_class_store = None
    selected_class_idx = None
    
    logger.info(f"All SAM3 classes: {[(c['label'], c['color']) for c in all_sam3_classes]}")
    
    for idx, sam3_class in enumerate(all_sam3_classes):
        if sam3_class["color"] == sam3_current_color:
            selected_class_store = sam3_class
            selected_class_idx = idx
            break
    
    if not selected_class_store:
        logger.error(f"No SAM3 class found with color: {sam3_current_color}")
        logger.error(f"Available colors: {[c['color'] for c in all_sam3_classes]}")
        notification = generate_notification(
            "SAM3 Error",
            "red",
            ANNOT_ICONS["parameters"],
            "No SAM3 class selected"
        )
        return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
    
    logger.info(f"Using SAM3 class: {selected_class_store['label']} with color {selected_class_store['color']}")
    
    # Load image
    try:
        image_idx_zero = image_idx - 1
        image_data = tiled_datasets.get_data_sequence_by_trimmed_uri(image_uri)[image_idx_zero]
        
        low = np.percentile(image_data.ravel(), 1)
        high = np.percentile(image_data.ravel(), 99)
        image_data = np.clip((image_data - low) / (high - low), 0, 1)
        image_data = (image_data * 255).astype(np.uint8)
        
        image_rgb = np.stack([image_data] * 3, axis=-1)
        pil_image = Image.fromarray(image_rgb)
        
        logger.info(f"Image loaded: {pil_image.size}")
        
    except Exception as e:
        logger.error(f"Image loading failed: {e}", exc_info=True)
        notification = generate_notification(
            "SAM3 Error",
            "red",
            ANNOT_ICONS["parameters"],
            f"Error loading image: {str(e)}"
        )
        return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
    
    # Run segmentation
    masks = None
    
    if active_mode == "text":
        if not text_prompt:
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                "Please enter a text prompt"
            )
            return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
        
        logger.info(f"Running text segmentation: '{text_prompt}'")
        try:
            masks = sam3_segmenter.segment_with_text(pil_image, text_prompt)
            logger.info(f"Text segmentation complete: {len(masks) if masks else 0} masks")
        except Exception as e:
            logger.error(f"Text segmentation failed: {e}", exc_info=True)
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                f"Text segmentation error: {str(e)}"
            )
            return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
    
    elif active_mode == "bbox":
        shapes = fig.get("layout", {}).get("shapes", [])
        logger.info(f"Total shapes in figure: {len(shapes)}")
        
        boxes = []
        for shape in shapes:
            # Only get rectangles marked as SAM3_bbox
            if (shape.get("type") == "rect" and 
                shape.get("label") == "SAM3_bbox"):
                bbox = [
                    float(shape["x0"]),
                    float(shape["y0"]),
                    float(shape["x1"]),
                    float(shape["y1"])
                ]
                boxes.append(bbox)
        
        logger.info(f"Extracted {len(boxes)} SAM3 bounding boxes: {boxes}")
        
        if len(boxes) == 0:
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                "Please draw bounding boxes first"
            )
            return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
        
        try:
            masks = sam3_segmenter.segment_with_boxes(pil_image, boxes)
            logger.info(f"Box segmentation complete: {len(masks) if masks else 0} masks")
        except Exception as e:
            logger.error(f"Box segmentation failed: {e}", exc_info=True)
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                f"Box segmentation error: {str(e)}"
            )
            return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
    
    elif active_mode == "point":
        if not points_data or "points" not in points_data or len(points_data["points"]) == 0:
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                "Please click points first"
            )
            return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
        
        points = points_data["points"]
        labels = points_data["labels"]
        logger.info(f"Running point segmentation: {len(points)} points")
        
        try:
            mask = sam3_segmenter.segment_with_points(pil_image, points, labels)
            masks = [mask] if mask is not None else None
            logger.info(f"Point segmentation complete: {1 if mask is not None else 0} masks")
        except Exception as e:
            logger.error(f"Point segmentation failed: {e}", exc_info=True)
            notification = generate_notification(
                "SAM3 Error",
                "red",
                ANNOT_ICONS["parameters"],
                f"Point segmentation error: {str(e)}"
            )
            return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
    
    if masks is None or (isinstance(masks, list) and len(masks) == 0):
        logger.warning("Segmentation returned no masks")
        notification = generate_notification(
            "SAM3 Error",
            "red",
            ANNOT_ICONS["parameters"],
            "SAM3 found no objects"
        )
        return notification, [no_update] * len(all_sam3_classes), no_update, no_update, no_update
    
    # Add masks to selected SAM3 class using SAM3 color
    class_id = selected_class_store["class_id"]
    class_color = selected_class_store["color"]
    image_idx_str = str(image_idx_zero)
    
    logger.info(f"Creating annotations with color: {class_color}")
    
    if image_idx_str not in selected_class_store["annotations"]:
        selected_class_store["annotations"][image_idx_str] = []
    
    num_masks_added = 0
    for mask in masks:
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        
        shape = convert_sam3_mask_to_annotation(mask, class_color, class_id)
        
        if shape:
            logger.info(f"Created shape with color: {shape['line']['color']}, fillcolor: {shape.get('fillcolor')}")
            selected_class_store["annotations"][image_idx_str].append(shape)
            num_masks_added += 1
    
    logger.info(f"Added {num_masks_added} SAM3 annotations to class {selected_class_store['label']} with color {class_color}")
    
    # Update all_sam3_classes with modified class
    all_sam3_classes[selected_class_idx] = selected_class_store
    
    # Update figure with SAM3 annotations only (remove temporary boxes/points)
    patched_fig = Patch()
    all_annotations = []
    for sam3_class in all_sam3_classes:
        if sam3_class["is_visible"] and image_idx_str in sam3_class["annotations"]:
            all_annotations += sam3_class["annotations"][image_idx_str]
    
    logger.info(f"Total annotations to display: {len(all_annotations)}")
    patched_fig["layout"]["shapes"] = all_annotations
    
    notification = generate_notification(
        "SAM3 Success",
        "green",
        ANNOT_ICONS["results"],
        f"Added {num_masks_added} auto-annotations"
    )
    
    cleared_bbox = {"boxes": []}
    cleared_points = {"points": [], "labels": []}
    
    return notification, all_sam3_classes, patched_fig, cleared_bbox, cleared_points


@callback(
    Output("sam3-points-store", "data", allow_duplicate=True),
    Output("image-viewer", "figure", allow_duplicate=True),
    Input("image-viewer", "clickData"),
    State("sam3-active-mode", "data"),
    State("sam3-points-store", "data"),
    State("sam3-current-class-selection", "data"),
    State("image-viewer", "figure"),
    prevent_initial_call=True,
)
def capture_sam3_points(click_data, active_mode, points_store, sam3_current_color, fig):
    """Capture points clicked in point mode"""
    
    if active_mode != "point":
        return no_update, no_update
    
    if click_data and "points" in click_data:
        point = click_data["points"][0]
        x, y = point["x"], point["y"]
        
        if points_store is None:
            points_store = {"points": [], "labels": []}
        
        points_store["points"].append([x, y])
        points_store["labels"].append(1)
        
        logger.info(f"Added SAM3 point at ({x}, {y})")
        
        patched_fig = Patch()
        existing_shapes = fig.get("layout", {}).get("shapes", [])
        
        marker = {
            "type": "circle",
            "xref": "x",
            "yref": "y",
            "x0": x - 5,
            "y0": y - 5,
            "x1": x + 5,
            "y1": y + 5,
            "line": {"color": sam3_current_color, "width": 2},
            "fillcolor": sam3_current_color,
            "opacity": 0.5,
            "editable": False,
            "label": "SAM3_point"
        }
        
        patched_fig["layout"]["shapes"] = existing_shapes + [marker]
        
        return points_store, patched_fig
    
    return no_update, no_update


@callback(
    Output("sam3-bbox-store", "data", allow_duplicate=True),
    Output("sam3-points-store", "data", allow_duplicate=True),
    Output("image-viewer", "figure", allow_duplicate=True),
    Input("sam3-clear-button", "n_clicks"),
    State("image-viewer", "figure"),
    State({"type": "sam3-class-store", "index": ALL}, "data"),
    State("image-selection-slider", "value"),
    prevent_initial_call=True,
)
def clear_sam3_prompts(n_clicks, fig, all_sam3_classes, image_idx):
    """Clear SAM3 prompts (boxes and points) but keep SAM3 annotations"""
    if n_clicks:
        logger.info("Clearing SAM3 prompts")
        patched_fig = Patch()
        
        # Get existing SAM3 annotations to preserve them
        image_idx_str = str(image_idx - 1)
        sam3_annotations = []
        for sam3_class in all_sam3_classes:
            if sam3_class["is_visible"] and image_idx_str in sam3_class["annotations"]:
                sam3_annotations += sam3_class["annotations"][image_idx_str]
        
        # Only keep SAM3 annotations, remove temporary drawn boxes and point markers
        patched_fig["layout"]["shapes"] = sam3_annotations
        
        return {"boxes": []}, {"points": [], "labels": []}, patched_fig
    return no_update, no_update, no_update