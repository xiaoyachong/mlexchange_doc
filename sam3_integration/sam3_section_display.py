import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify


def sam3_annotation_section():
    """
    Create the SAM3 auto-annotation section for the sidebar
    """
    return html.Div([
        dmc.Space(h=5),
        dmc.Text(
            "SAM3 Auto-Annotation",
            size="md",
            weight=500,
            color="#00313C"
        ),
        dmc.Space(h=10),
        dmc.Text(
            "Use AI to automatically segment objects",
            size="xs",
            color="#9EA4AB"
        ),
        dmc.Space(h=15),
        
        # Prompt type selector
        dmc.Select(
            id="sam3-prompt-type",
            label="Prompt Type",
            placeholder="Select prompt type...",
            data=[
                {"value": "text", "label": "Text Description"},
                {"value": "bbox", "label": "Bounding Box"},
                {"value": "point", "label": "Point Clicks"},
            ],
            value=None,
            clearable=True,
        ),
        dmc.Space(h=15),
        
        # Text prompt input (shown for text mode)
        dmc.TextInput(
            id="sam3-prompt-input",
            label="Text Prompt",
            placeholder='e.g., "sand particles"',
            disabled=True,
        ),
        dmc.Space(h=15),
        
        # Point type selector (shown for point mode)
        dmc.RadioGroup(
            id="sam3-point-type",
            label="Point Type",
            children=[
                dmc.Radio("Positive (include)", value="positive"),
                dmc.Radio("Negative (exclude)", value="negative"),
            ],
            value="positive",
            size="sm",
        ),
        dmc.Space(h=15),
        
        # Instructions text
        html.Div(
            id="sam3-instructions",
            children=[
                dmc.Text(
                    "1. Select prompt type above",
                    size="xs",
                    color="#9EA4AB"
                ),
                dmc.Text(
                    "2. For text: enter description",
                    size="xs",
                    color="#9EA4AB"
                ),
                dmc.Text(
                    "3. For bbox: draw rectangles on image",
                    size="xs",
                    color="#9EA4AB"
                ),
                dmc.Text(
                    "4. For points: click on image",
                    size="xs",
                    color="#9EA4AB"
                ),
                dmc.Text(
                    "5. Click 'Generate' button",
                    size="xs",
                    color="#9EA4AB"
                ),
            ]
        ),
        dmc.Space(h=15),
        
        # Action buttons
        dmc.Group([
            dmc.Button(
                "Generate",
                id="sam3-run-button",
                variant="light",
                color="indigo",
                style={"flex": "1"},
            ),
            dmc.ActionIcon(
                DashIconify(icon="mdi:delete", width=20),
                id="sam3-clear-prompts",
                variant="light",
                color="red",
                size="lg",
            ),
        ], spacing="xs"),
        
        dmc.Space(h=10),
        
        # Info alert
        dmc.Alert(
            children=[
                dmc.Text(
                    "SAM3 generates pixel-level masks that are automatically added to the current class.",
                    size="xs",
                )
            ],
            title="Note",
            color="blue",
            icon=DashIconify(icon="mdi:information"),
        ),
        
        # Hidden stores for bbox and point data
        dcc.Store(id="sam3-bbox-store", data={"boxes": []}),
        dcc.Store(id="sam3-points-store", data={"points": [], "labels": []}),
        dcc.Store(id="sam3-bbox-mode", data=False),
        dcc.Store(id="sam3-point-mode", data=False),
    ])
