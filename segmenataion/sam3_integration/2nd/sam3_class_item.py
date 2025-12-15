import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify


def get_sam3_action_icon(type, class_id, icon):
    """Returns action icons for SAM3 class"""
    return dmc.ActionIcon(
        id={
            "type": type,
            "index": class_id,
        },
        variant="subtle",
        color="gray",
        children=DashIconify(icon=icon),
        size="lg",
    )


def sam3_class_item(class_color, class_label, existing_ids, data=None):
    """
    Returns the layout for a SAM3 class item - separate from annotation classes
    """
    if data:
        class_color = data["color"]
        class_label = data["label"]
        class_id = data["class_id"]
        annotations = data["annotations"]
        is_visible = data["is_visible"]
    else:
        class_id = 0 if not existing_ids else max(existing_ids) + 1
        annotations = {}
        is_visible = True
    
    class_color_transparent = class_color + "50"

    return html.Div(
        [
            # Store for SAM3 class data
            dcc.Store(
                id={
                    "type": "sam3-class-store",
                    "index": class_id,
                },
                data={
                    "annotations": annotations,
                    "color": class_color,
                    "label": class_label,
                    "is_visible": is_visible,
                    "class_id": class_id,
                },
            ),
            # Stores for triggering callbacks
            dcc.Store(id={"type": "sam3-deleted-class-store", "index": class_id}),
            dcc.Store(
                id={"type": "sam3-hide-show-class-store", "index": class_id},
                data={"is_visible": True},
            ),
            dcc.Store(
                id={"type": "sam3-edit-class-store", "index": class_id},
                data=False,
            ),
            html.Div(
                [
                    # Colored box
                    html.Div(
                        style={
                            "width": "25px",
                            "height": "25px",
                            "backgroundColor": class_color_transparent,
                            "margin": "5px",
                            "borderRadius": "3px",
                            "border": f"2px solid {class_color}",
                        },
                        id={
                            "type": "sam3-class-color",
                            "index": class_id,
                        },
                    ),
                    html.Div(
                        class_label,
                        id={
                            "type": "sam3-class-label",
                            "index": class_id,
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "flex-row",
                    "alignItems": "center",
                    "color": "#9EA4AB",
                },
            ),
            html.Div(
                [
                    get_sam3_action_icon("sam3-hide-class", class_id, "mdi:eye"),
                    get_sam3_action_icon("sam3-edit-class", class_id, "uil:edit"),
                    get_sam3_action_icon("sam3-delete-class", class_id, "octicon:trash-24"),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "flex-row",
                    "alignItems": "center",
                    "padding": "3px",
                },
            ),
            # Edit modal
            dmc.Modal(
                id={"type": "sam3-edit-class-modal", "index": class_id},
                title="Edit SAM3 Class",
                children=[
                    html.Div(
                        [
                            dbc.Input(
                                type="color",
                                id={
                                    "type": "sam3-edit-class-colorpicker",
                                    "index": class_id,
                                },
                                style={"width": 75, "height": 50},
                            ),
                            dmc.Space(w=25),
                            html.Div(
                                [
                                    dmc.TextInput(
                                        id={
                                            "type": "sam3-edit-class-text-input",
                                            "index": class_id,
                                        },
                                        placeholder="New class label...",
                                    ),
                                    html.Div(
                                        id={
                                            "type": "sam3-bad-edit-label",
                                            "index": class_id,
                                        },
                                        style={
                                            "color": "red",
                                            "fontSize": "12px",
                                            "padding": "3px",
                                        },
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "flex-row",
                            "alignItems": "center",
                        },
                    ),
                    html.Div(
                        [
                            dmc.Button(
                                id={
                                    "type": "sam3-save-edited-class-btn",
                                    "index": class_id,
                                },
                                children="Save",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "flex-end",
                        },
                    ),
                ],
            ),
            # Delete modal
            dmc.Modal(
                id={"type": "sam3-delete-class-modal", "index": class_id},
                children=[
                    dmc.Center(
                        dmc.Text(
                            "This action will permanently clear all SAM3 annotations from this class. Are you sure?",
                        )
                    ),
                    dmc.Space(h=10),
                    html.Div(
                        [
                            dmc.Button(
                                id={
                                    "type": "sam3-cancel-delete-class-btn",
                                    "index": class_id,
                                },
                                children="Cancel",
                            ),
                            dmc.Space(w=10),
                            dmc.Button(
                                id={
                                    "type": "sam3-confirm-delete-class-btn",
                                    "index": class_id,
                                },
                                children="Confirm",
                                variant="outline",
                                color="red",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "flex-end",
                        },
                    ),
                ],
            ),
            # Cannot delete last class modal
            dmc.Modal(
                id={"type": "sam3-cannot-delete-last-class-modal", "index": class_id},
                children=[
                    dmc.Center(
                        dmc.Text("You cannot delete the last class"),
                    ),
                    dmc.Space(h=10),
                    html.Div(
                        [
                            dmc.Button(
                                id={
                                    "type": "sam3-ok-not-delete-last-class-btn",
                                    "index": class_id,
                                },
                                children="Ok",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "flex-end",
                        },
                    ),
                ],
            ),
        ],
        style={
            "border": "1px solid #EAECEF",
            "borderRadius": "3px",
            "marginBottom": "4px",
            "display": "flex",
            "justifyContent": "space-between",
        },
        className="sam3-class",
        id={"type": "sam3-class", "index": class_id},
    )