# Switching from Local Filesystem to Tiled Server Only

## Overview
This guide explains how to configure your Latent Space Explorer application to use **only Tiled server data** instead of local filesystem data.

## Key Decision: Keep the File Manager

**Recommendation: Keep the File Manager component** - it already supports Tiled and will adapt automatically.

---

## Required Code Changes

### 1. Update `src/app_layout.py`

#### Current Code:
```python
dash_file_explorer = FileManager(
    READ_DIR,  # ← This points to local filesystem
    open_explorer=False,
    api_key=DATA_TILED_KEY,
)
```

#### Change To:
```python
# Add this import at the top
DATA_TILED_URI = os.getenv("DATA_TILED_URI", "http://tiled:8000")

# Update FileManager initialization
dash_file_explorer = FileManager(
    DATA_TILED_URI,  # ← Point directly to Tiled server
    open_explorer=False,
    api_key=DATA_TILED_KEY,
)
```

### 2. Update Environment Variables

In your `.env` file:

```bash
# Remove or comment out local filesystem config:
# READ_DIR=data
# READ_DIR_MOUNT=/path/to/data

# Keep only Tiled configuration:
DATA_TILED_URI=http://your-tiled-server:8000
DATA_TILED_KEY=your-tiled-api-key
RESULTS_TILED_URI=http://your-tiled-server:8000
RESULTS_TILED_API_KEY=your-results-api-key
```

### 3. Clean Up Unused Variables (Optional)

In `frontend.py`, remove these lines:

```python
# Remove:
READ_DIR = os.getenv("READ_DIR", "data")
READ_DIR_MOUNT = os.getenv("READ_DIR_MOUNT", None)
```

---

## UI Modifications (Optional)

### Option 1: No Changes (Recommended)
Keep the UI as-is. The file manager adapts automatically to show Tiled structure.

### Option 2: Update Label for Clarity
Make it explicit that you're browsing Tiled data.

In `src/components/sidebar.py`:

```python
dbc.AccordionItem(
    id="data-selection-controls",
    title="Data Selection (Tiled Server)",  # ← Changed
    children=file_explorer,
),
```

### Option 3: Add Server Badge
Add a visual indicator of the data source.

```python
dbc.AccordionItem(
    id="data-selection-controls",
    title=[
        "Data Selection ",
        dbc.Badge("Tiled", color="info", className="ms-2")
    ],
    children=file_explorer,
),
```

### Option 4: Add Connection Info
Show which Tiled server is connected (useful for debugging).

In `src/components/sidebar.py`, at the top of the sidebar:

```python
def sidebar(file_explorer, job_manager, clustering_job_manager):
    sidebar = html.Div([
        dbc.Offcanvas(
            id="sidebar-offcanvas",
            is_open=True,
            backdrop=False,
            scrollable=True,
            style={
                "padding": "80px 0px 0px 0px",
                "width": "500px",
            },
            title="Controls",
            children=[
                # Add connection info
                dbc.Alert(
                    [
                        DashIconify(icon="mdi:server-network", className="me-2"),
                        f"Connected to: {os.getenv('DATA_TILED_URI', 'Tiled Server')}"
                    ],
                    color="info",
                    className="mb-3"
                ),
                
                dbc.Accordion(
                    # ... rest of your accordion code
```

---

## Verification Steps

### 1. Check Configuration
Add logging to verify the setup:

```python
# In src/app_layout.py, after creating dash_file_explorer:
import logging
logger = logging.getLogger(__name__)

logger.info(f"File Manager configured for: {DATA_TILED_URI}")
logger.info(f"Using API key: {'Yes' if DATA_TILED_KEY else 'No'}")
```

### 2. Test Connection
Run the application and check:
- File manager loads without errors
- You can browse Tiled containers/collections
- You can select datasets
- Data preview works

### 3. Verify Data Loading
In the callbacks that use data:

```python
# In src/callbacks/display.py - these should work unchanged
data_project = DataProject.from_dict(data_project_dict, api_key=DATA_TILED_KEY)
selected_images, _ = data_project.read_datasets(
    selected_indices,
    resize=True,
    export="pillow",
    log=log_transform,
    percentiles=percentiles,
)
```

---

## What Changes for Users

### Before (Local Files):
```
📁 data/
  📁 experiment_1/
    🖼️ image_001.tif
    🖼️ image_002.tif
  📁 experiment_2/
    🖼️ image_003.tif
```

### After (Tiled):
```
📊 my_tiled_container/
  📊 experiment_1/
    🔢 image_array [shape: (512, 512, 100)]
  📊 experiment_2/
    🔢 processed_data [shape: (256, 256, 50)]
```

**Navigation experience remains the same** - just different data structures displayed.

---

## What You Keep

✅ Data browsing UI in the sidebar  
✅ Data selection functionality  
✅ `DataProject` abstraction layer  
✅ All existing callbacks (no changes needed)  
✅ Consistent user experience  
✅ Live mode functionality  
✅ Experiment replay functionality  

## What Changes

❌ No more local filesystem access  
✅ All data comes from Tiled server  
✅ Cleaner deployment (no need to mount local directories)  
✅ Better scalability (Tiled can serve large datasets efficiently)  
✅ Multi-user support (Tiled handles concurrent access)  

---

## Architecture Benefits

### Why Keep File Manager:
1. **Already Supports Tiled**: The `FileManager` library is designed to work with both local and remote data sources
2. **No Code Rewrite**: Changing the URI is a configuration change, not an architectural change
3. **Consistent Interface**: Users get the same browsing experience
4. **Future Flexibility**: Easy to switch back or support multiple sources if needed

### How File Manager Adapts:
- Detects Tiled URI automatically (starts with `http://` or `https://`)
- Uses Tiled client library internally
- Handles authentication via `api_key` parameter
- Renders Tiled-specific data structures (containers, arrays, dataframes)

---

## Troubleshooting

### Issue: File Manager Shows Empty
**Solution**: Check that:
- `DATA_TILED_URI` is correct and accessible
- `DATA_TILED_KEY` has proper permissions
- Tiled server is running and responsive

### Issue: Can't Load Images/Data
**Solution**: Verify:
- Data exists in Tiled at the expected paths
- API key has read permissions
- Tiled server isn't rate-limiting requests

### Issue: Slow Data Loading
**Solution**: Consider:
- Using Tiled's built-in caching
- Adjusting `httpx.Timeout` values in `data_utils.py`
- Enabling Tiled server-side compression

---

## Summary

**Minimal changes required:**
1. Change `FileManager` initialization to use Tiled URI instead of local path
2. Update environment variables
3. Optionally update UI labels for clarity

**No need to:**
- Remove the file manager component
- Modify callbacks
- Change data loading logic
- Rewrite the architecture

The file manager is **data-source agnostic** by design. You're just pointing it to a different data source.
