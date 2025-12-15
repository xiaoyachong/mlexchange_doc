import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from components.sam3_class_item import sam3_class_item
from constants import ANNOT_ICONS


def sam3_annotation_section():
    """
    Create the SAM3 auto-annotation section with independent class management
    """
    return html.Div([
        dmc.Text(
            "SAM3 Toolbar",
            size="sm",
            align="right",
            color="#9EA4AB",
        ),
        dmc.Space(h=15),
        
        # Toolbar
        dmc.Grid(
            [
                dmc.Space(w=8),
                html.Div(
                    children=[
                        dmc.Tooltip(
                            label="Pan and zoom",
                            withArrow=True,
                            position="top",
                            color="#464646",
                            children=dmc.ActionIcon(
                                id="sam3-pan-zoom",
                                variant="subtle",
                                color="gray",
                                children=DashIconify(
                                    icon=ANNOT_ICONS["pan-and-zoom"],
                                    width=20,
                                ),
                                size="lg",
                            ),
                        ),
                    ],
                    className="flex-row",
                    style={
                        "justifyContent": "space-evenly",
                        "padding": "2.5px",
                        "border": "1px solid #EAECEF",
                        "borderRadius": "5px",
                    },
                ),
                dmc.Space(w=10),
                html.Div(
                    children=[
                        dmc.Tooltip(
                            label="Text prompt",
                            withArrow=True,
                            position="top",
                            color="#464646",
                            children=dmc.ActionIcon(
                                id="sam3-text-mode",
                                variant="subtle",
                                color="gray",
                                children=DashIconify(
                                    icon="mdi:text-box",
                                    width=20,
                                ),
                                style={"backgroundColor": "#EAECEF"},
                                size="lg",
                            ),
                        ),
                        dmc.Tooltip(
                            label="Bounding box",
                            withArrow=True,
                            position="top",
                            color="#464646",
                            children=dmc.ActionIcon(
                                id="sam3-bbox-mode",
                                variant="subtle",
                                color="gray",
                                children=DashIconify(
                                    icon=ANNOT_ICONS["rectangle"],
                                    width=20,
                                ),
                                size="lg",
                            ),
                        ),
                        dmc.Tooltip(
                            label="Point click",
                            withArrow=True,
                            position="top",
                            color="#464646",
                            children=dmc.ActionIcon(
                                id="sam3-point-mode",
                                variant="subtle",
                                color="gray",
                                children=DashIconify(
                                    icon="mdi:crosshairs-gps",
                                    width=20,
                                ),
                                size="lg",
                            ),
                        ),
                    ],
                    className="flex-row",
                    style={
                        "width": "301px",
                        "justifyContent": "space-evenly",
                        "padding": "2.5px",
                        "border": "1px solid #EAECEF",
                        "borderRadius": "5px",
                    },
                ),
            ]
        ),
        dmc.Space(h=10),
        
        # Text input
        html.Div(
            id="sam3-text-input-container",
            children=[
                dmc.TextInput(
                    id="sam3-text-prompt",
                    placeholder='e.g., "particles"',
                    style={"width": "100%"},
                ),
            ],
            style={"display": "block"}
        ),
        dmc.Space(h=20),
        
        # SAM3 Class Management Section
        html.Div([
            dmc.Text(
                "SAM3 Classes",
                size="sm",
                align="right",
                color="#9EA4AB",
            ),
            dmc.Space(h=10),
            html.Div(
                children=[
                    sam3_class_item("#4169E1", "SAM3 Class 1", [])
                ],
                id="sam3-class-container",
            ),
            dmc.Button(
                "+ Add new class... ",
                id="sam3-generate-class",
                variant="outline",
                style={"width": "100%"},
                className="add-class-btn",
            ),
            dcc.Store(id="sam3-current-class-selection", data="#4169E1"),
            dmc.Space(h=20),
        ]),
        
        # Modal for creating new SAM3 class
        dmc.Modal(
            id="sam3-generate-class-modal",
            title="Create a new SAM3 class",
            children=[
                html.Div(
                    [
                        dmc.TextInput(
                            id="sam3-class-label",
                            placeholder="Class label...",
                            style={"width": "100%"},
                        ),
                        html.Div(
                            id="sam3-bad-label-color",
                            style={
                                "color": "red",
                                "fontSize": "12px",
                                "padding": "3px",
                            },
                        ),
                    ]
                ),
                dmc.Space(h=10),
                html.Div(
                    [
                        dmc.Button(
                            id="sam3-create-class",
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
        
        # Generate button
        dmc.Button(
            "Generate SAM3",
            id="sam3-generate-button",
            variant="filled",
            color="indigo",
            style={"width": "100%"},
        ),
        dmc.Space(h=3),
        
        # Clear button
        dmc.Button(
            "Clear prompts",
            id="sam3-clear-button",
            variant="outline",
            style={"width": "100%"},
        ),
        dmc.Space(h=20),
        
        # Hidden stores
        dcc.Store(id="sam3-active-mode", data="text"),
        dcc.Store(id="sam3-bbox-store", data={"boxes": []}),
        dcc.Store(id="sam3-points-store", data={"points": [], "labels": []}),
    ])