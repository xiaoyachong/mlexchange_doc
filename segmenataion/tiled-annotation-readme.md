# Save/Load Annotation Metadata via Tiled

## Overview
This describes the minimal changes needed to replace the flat `exported_annotation_data.json` file with Tiled for saving and loading annotation metadata. The pixel masks are already saved to Tiled — this extends that to also cover the raw annotation coordinates.

The approach stores annotation JSON as Tiled container metadata alongside a dummy array, reusing the existing `TiledMaskHandler` class.

---

## 1. Changes to `utils/data_utils.py`

Add two new methods to the `TiledMaskHandler` class:

```python
def save_annotation_metadata(self, all_annotations, image_uri, user):
    """
    Saves raw annotation metadata to Tiled under:
    /<user>/annotations/<image_uri>/<timestamp>
    """
    import time
    import numpy as np

    container_keys = [user, "annotations"] + image_uri.strip("/").split("/")
    last_container = self.mask_client
    for key in container_keys:
        if key not in last_container.keys():
            last_container = last_container.create_container(key=key)
        else:
            last_container = last_container[key]

    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S")
    metadata = {
        "user": user,
        "source": image_uri,
        "time": timestamp,
        "data": all_annotations,
    }
    # Store a dummy array to attach metadata to
    dummy = np.array([0])
    last_container.write_array(
        key=timestamp,
        array=dummy,
        metadata=metadata,
    )
    return timestamp


def load_annotation_metadata(self, image_uri, user):
    """
    Loads all saved annotation metadata for a given user and image_uri from Tiled.
    Returns a list sorted by timestamp (latest first).
    """
    try:
        container_keys = [user, "annotations"] + image_uri.strip("/").split("/")
        last_container = self.mask_client
        for key in container_keys:
            last_container = last_container[key]
    except KeyError:
        return []

    results = []
    for key in last_container.keys():
        meta = last_container[key].metadata
        results.append(meta)
    return sorted(results, key=lambda x: x["time"], reverse=True)
```

---

## 2. Changes to `callbacks/control_bar.py`

### 2a. Replace `save_data` callback

**Before:**
```python
@callback(
    Output("data-modal-save-status", "children"),
    Input("save-annotations", "n_clicks"),
    State("annotation-store", "data"),
    State({"type": "annotation-class-store", "index": ALL}, "data"),
    State("image-uri", "value"),
    prevent_initial_call=True,
)
def save_data(n_clicks, global_store, all_annotations, image_uri):
    if not n_clicks:
        raise PreventUpdate
    if all_annotations:
        export_data = {
            "user": USER_NAME,
            "source": image_uri,
            "time": time.strftime("%Y-%m-%d-%H:%M:%S"),
            "data": json.dumps(all_annotations),
        }
        export_data_json = json.dumps(export_data)
        if export_data["data"] != "{}":
            with open(EXPORT_FILE_PATH, "a+") as f:
                f.write(export_data_json + "\n")
        return "Data saved!"
    return "No annotations to save!"
```

**After:**
```python
@callback(
    Output("data-modal-save-status", "children"),
    Input("save-annotations", "n_clicks"),
    State("annotation-store", "data"),
    State({"type": "annotation-class-store", "index": ALL}, "data"),
    State("image-uri", "value"),
    prevent_initial_call=True,
)
def save_data(n_clicks, global_store, all_annotations, image_uri):
    if not n_clicks:
        raise PreventUpdate
    if all_annotations:
        timestamp = tiled_masks.save_annotation_metadata(
            all_annotations, image_uri, USER_NAME
        )
        return f"Data saved at {timestamp}!"
    return "No annotations to save!"
```

### 2b. Replace `populate_load_annotations_dropdown_menu_options` callback

**Before:**
```python
def populate_load_annotations_dropdown_menu_options(modal_opened, image_uri):
    data = tiled_masks.DEV_load_exported_json_data(
        EXPORT_FILE_PATH, USER_NAME, image_uri
    )
    ...
```

**After:**
```python
def populate_load_annotations_dropdown_menu_options(modal_opened, image_uri):
    data = tiled_masks.load_annotation_metadata(image_uri, USER_NAME)
    ...
```

### 2c. Replace `load_and_apply_selected_annotations` callback

**Before:**
```python
def load_and_apply_selected_annotations(selected_annotation, image_uri, img_idx):
    ...
    data = tiled_masks.DEV_load_exported_json_data(
        EXPORT_FILE_PATH, USER_NAME, image_uri
    )
    data = tiled_masks.DEV_filter_json_data_by_timestamp(
        data, str(selected_annotation_timestamp)
    )
    data = data[0]["data"]
    ...
```

**After:**
```python
def load_and_apply_selected_annotations(selected_annotation, image_uri, img_idx):
    ...
    all_data = tiled_masks.load_annotation_metadata(image_uri, USER_NAME)
    data = [d for d in all_data if d["time"] == str(selected_annotation_timestamp)]
    data = data[0]["data"]
    ...
```

---

## 3. Tiled Storage Structure

Annotations will be stored in Tiled under:
```
tiled/
└── <USER_NAME>/
    └── annotations/
        └── <image_uri>/
            ├── 2024-02-18-10:00:00   ← dummy array + annotation metadata
            ├── 2024-02-18-11:00:00
            └── 2024-02-18-12:00:00
```

---

## 4. Migrate Existing Annotations from JSON file

Before removing the old JSON file, migrate existing data to Tiled:

```python
import json
from utils.data_utils import tiled_masks

with open("exported_annotation_data.json", "r") as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            entry["data"] = json.loads(entry["data"])
            tiled_masks.save_annotation_metadata(
                entry["data"], entry["source"], entry["user"]
            )
print("Migration complete.")
```

Run once:
```bash
docker exec -it mlex_tomo_framework-mlex_segmentation-1 python migrate_annotations.py
```

---

## 5. Cleanup (optional)

Once migration is verified:
- Remove `EXPORT_FILE_PATH` environment variable
- Remove file-related code (`open`, `json.dumps`, `DEV_load_exported_json_data`, `DEV_filter_json_data_by_timestamp`) from `data_utils.py` and `control_bar.py`
- Remove the volume mount for the JSON file if added

---

## Notes
- This reuses the existing `MASK_TILED_URI` and `MASK_TILED_API_KEY` — no new infrastructure needed
- Annotation metadata is stored as Tiled container metadata alongside a dummy array (Tiled requires an array to attach metadata to)
- This is a pragmatic workaround — Tiled is designed for arrays, not JSON metadata. A proper PostgreSQL implementation would be cleaner long-term
