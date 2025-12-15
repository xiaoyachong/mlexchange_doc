# Tiled Live Mode Code Explanation

This document provides a detailed line-by-line explanation of how the Latent Space Explorer processes Tiled URLs in live mode. The code handles WebSocket messages containing feature vectors, extracts metadata from URLs, and manages data tracking.

## Example Input

For this explanation, we'll assume we have a message with:

```javascript
data = {
  tiled_url: "https://tiled-demo.blueskyproject.io/api/v1/metadata/rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image?slice=32",
  index: "32",
  feature_vector: [0.32375478744506836, 3.48423433303833]
}
```

And the existing state:

```javascript
buffer_data = [/* existing entries */];
data_project_dict = {
  "root_uri": "",
  "data_type": "",
  "datasets": [],
  "project_id": "live"
};
live_indices = [5, 12, 18, 24];
```

## Line-by-Line Explanation

### 1. Update Buffer Data

```javascript
buffer_data = [...buffer_data, new_entry];
```

- **Purpose**: Adds a new entry to the buffer data array
- **Values**:
  - `buffer_data` (before): `[/* existing entries */]`
  - `new_entry`: `{ "feature_vector": [0.32375478744506836, 3.48423433303833], "num_components": 2 }`
  - `buffer_data` (after): `[/* existing entries */, { "feature_vector": [0.32375478744506836, 3.48423433303833], "num_components": 2 }]`

### 2. Log Buffer Data

```javascript
log.debug("Updated buffer_data:", buffer_data);
```

- **Purpose**: Logs the updated buffer data
- **Values**:
  - `buffer_data`: `[/* existing entries */, { "feature_vector": [0.32375478744506836, 3.48423433303833], "num_components": 2 }]`

### 3. Extract Tiled URL

```javascript
let tiled_url = data.tiled_url;
```

- **Purpose**: Extracts the tiled URL from the message data
- **Values**:
  - `data.tiled_url`: `"https://tiled-demo.blueskyproject.io/api/v1/metadata/rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image?slice=32"`
  - `tiled_url`: `"https://tiled-demo.blueskyproject.io/api/v1/metadata/rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image?slice=32"`

### 4. Parse Index

```javascript
let index = parseInt(data.index);
```

- **Purpose**: Parses the index from the message data
- **Values**:
  - `data.index`: `"32"`
  - `index`: `32`

### 5. Log URL and Index

```javascript
log.debug("Tiled URI:", tiled_url, "Index:", index);
```

- **Purpose**: Logs the tiled URL and index
- **Values**:
  - `tiled_url`: `"https://tiled-demo.blueskyproject.io/api/v1/metadata/rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image?slice=32"`
  - `index`: `32`

### 6. Create URL Object

```javascript
let url = new URL(tiled_url);
```

- **Purpose**: Creates a URL object from the tiled URL
- **Values**:
  - `tiled_url`: `"https://tiled-demo.blueskyproject.io/api/v1/metadata/rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image?slice=32"`
  - `url`: URL object with properties like protocol, host, pathname, search, etc.

### 7. Split Path into Parts

```javascript
const path_parts = url.pathname.split('/').filter(p => p !== '');
```

- **Purpose**: Splits the pathname into parts and filters out empty strings
- **Values**:
  - `url.pathname`: `"/api/v1/metadata/rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image"`
  - `path_parts`: `["api", "v1", "metadata", "rsoxs", "raw", "468810ed-2ff9-4e92-8ca9-dcb376d01a56", "primary", "data", "Small Angle CCD Detector_image"]`

### 8. Initialize Root URI

```javascript
let root_uri = url.origin;
```

- **Purpose**: Extracts the origin (protocol, hostname, port) from the URL
- **Values**:
  - `url.origin`: `"https://tiled-demo.blueskyproject.io"`
  - `root_uri`: `"https://tiled-demo.blueskyproject.io"`

### 9. Initialize URI

```javascript
let uri = "";
```

- **Purpose**: Initializes the URI variable
- **Values**:
  - `uri`: `""`

### 10-14. Find API Index

```javascript
const apiIndex = path_parts.findIndex((p, i) =>
    p === 'api' &&
    path_parts[i + 1] === 'v1' &&
    ['metadata', 'array'].includes(path_parts[i + 2])
);
```

- **Purpose**: Finds the index of 'api' in the path parts, where it's followed by 'v1' and then 'metadata' or 'array'
- **Values**:
  - `path_parts`: `["api", "v1", "metadata", "rsoxs", "raw", "468810ed-2ff9-4e92-8ca9-dcb376d01a56", "primary", "data", "Small Angle CCD Detector_image"]`
  - `apiIndex`: `0` (because 'api' is at index 0, followed by 'v1' at index 1 and 'metadata' at index 2)

### 15-24. Extract Base URI and Dataset URI

```javascript
if (apiIndex !== -1) {
    const base_root_parts = path_parts.slice(0, apiIndex + 3);
    root_uri = `${url.protocol}//${url.host}/${base_root_parts.join('/')}`;
    uri = decodeURIComponent(path_parts.slice(apiIndex + 3).join('/'));
} else {
    console.warn("Unexpected Tiled URL format:", tiled_url);
}
```

- **Purpose**: If the API index is found, it extracts the base URI and the dataset URI
- **Values**:
  - `apiIndex`: `0`
  - `base_root_parts`: `["api", "v1", "metadata"]`
  - `url.protocol`: `"https:"`
  - `url.host`: `"tiled-demo.blueskyproject.io"`
  - `root_uri` (updated): `"https://tiled-demo.blueskyproject.io/api/v1/metadata"`
  - `path_parts.slice(apiIndex + 3)`: `["rsoxs", "raw", "468810ed-2ff9-4e92-8ca9-dcb376d01a56", "primary", "data", "Small Angle CCD Detector_image"]`
  - `uri`: `"rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image"`

### 25-26. Extract Query Parameters

```javascript
const params = new URLSearchParams(url.search);
const sliceParam = params.get('slice');
```

- **Purpose**: Extracts the query parameters and gets the 'slice' parameter
- **Values**:
  - `url.search`: `"?slice=32"`
  - `params`: URLSearchParams object with one key-value pair: 'slice' -> '32'
  - `sliceParam`: `"32"`

### 27-33. Parse Slice Parameter

```javascript
if (sliceParam) {
    const sliceParts = sliceParam.split(',');
    const parsedIndex = parseInt(sliceParts[0], 10);
    if (!isNaN(parsedIndex)) {
        index = parsedIndex;
    }
}
```

- **Purpose**: If there's a slice parameter, it parses it to get the index
- **Values**:
  - `sliceParam`: `"32"`
  - `sliceParts`: `["32"]`
  - `parsedIndex`: `32`
  - `index` (unchanged): `32`

### 34-41. Fallback to Dataset Matching

```javascript
else {
    const match = data_project_dict.datasets.find(d => d.uri === uri);
    if (match) {
        index = match.cumulative_data_count;
    } else {
        console.warn(`No matching dataset entry for uri: ${uri}`);
    }
}
```

- **Purpose**: If there's no slice parameter, it tries to find a matching dataset and uses its cumulative count as the index
- **Values**:
  - This block is skipped because `sliceParam` is not empty

### 42. Log Extracted Information

```javascript
log.debug("Root URI:", root_uri, "URI:", uri, "Index:", index);
```

- **Purpose**: Logs the root URI, URI, and index
- **Values**:
  - `root_uri`: `"https://tiled-demo.blueskyproject.io/api/v1/metadata"`
  - `uri`: `"rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image"`
  - `index`: `32`

### 43-46. Update Live Indices

```javascript
if (index >= 0) {
    live_indices = [...live_indices, index];
    log.debug("Updated live_indices:", live_indices);
}
```

- **Purpose**: If the index is valid, it adds it to the live indices array
- **Values**:
  - `index`: `32`
  - `live_indices` (before): `[5, 12, 18, 24]`
  - `live_indices` (after): `[5, 12, 18, 24, 32]`

### 47-48. Calculate Cumulative Size

```javascript
let cum_size = Math.max(...live_indices) + 1;
log.debug("Cumulative size:", cum_size);
```

- **Purpose**: Calculates the cumulative size as the maximum index + 1
- **Values**:
  - `live_indices`: `[5, 12, 18, 24, 32]`
  - `Math.max(...live_indices)`: `32`
  - `cum_size`: `33`

### 49-56. Update Data Project Dictionary Root URI

```javascript
if (data_project_dict["root_uri"] !== root_uri) {
    data_project_dict = {
        ...data_project_dict,
        "root_uri": root_uri,
        "data_type": "tiled"
    };
    log.info("Updated data_project_dict root_uri and data_type:", data_project_dict);
}
```

- **Purpose**: If the root URI has changed, it updates the data project dictionary
- **Values**:
  - `data_project_dict["root_uri"]`: `""`
  - `root_uri`: `"https://tiled-demo.blueskyproject.io/api/v1/metadata"`
  - `data_project_dict` (after): 
    ```javascript
    {
      "root_uri": "https://tiled-demo.blueskyproject.io/api/v1/metadata",
      "data_type": "tiled",
      "datasets": [],
      "project_id": "live"
    }
    ```

### 57-67. Initialize Datasets

```javascript
if (data_project_dict["datasets"].length === 0) {
    data_project_dict = {
        ...data_project_dict,
        "datasets": [{
            "uri": uri,
            "cumulative_data_count": cum_size
        }]
    };
    log.debug("Initialized datasets in data_project_dict:", data_project_dict["datasets"]);
}
```

- **Purpose**: If there are no datasets, it initializes the datasets array with one entry
- **Values**:
  - `data_project_dict["datasets"].length`: `0`
  - `uri`: `"rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image"`
  - `cum_size`: `33`
  - `data_project_dict` (after): 
    ```javascript
    {
      "root_uri": "https://tiled-demo.blueskyproject.io/api/v1/metadata",
      "data_type": "tiled",
      "datasets": [{
        "uri": "rsoxs/raw/468810ed-2ff9-4e92-8ca9-dcb376d01a56/primary/data/Small Angle CCD Detector_image",
        "cumulative_data_count": 33
      }],
      "project_id": "live"
    }
    ```

### 68-81. Append to Datasets

```javascript
else {
    data_project_dict = {
        ...data_project_dict,
        "datasets": [
            ...data_project_dict["datasets"],
            {
                "uri": uri,
                "cumulative_data_count": cum_size
            }
        ]
    };
    log.debug("Appended to datasets in data_project_dict:", data_project_dict["datasets"]);
}
```

- **Purpose**: If there are already datasets, it appends a new one
- **Values**:
  - This block is skipped because `data_project_dict["datasets"].length` is 0

## Summary

This code:

1. Adds a new entry to the buffer data
2. Extracts the Tiled URL and index from the message
3. Parses the URL to get the root URI and dataset URI
4. Extracts the slice parameter from the URL and updates the index if needed
5. Adds the index to the live indices array
6. Calculates the cumulative size as the maximum index + 1
7. Updates the data project dictionary with the new URI and cumulative size

## Key Variables and Their Purpose

- **buffer_data**: Stores the feature vectors and related information
- **tiled_url**: The raw URL string from the message
- **url**: JavaScript URL object that makes it easier to parse the URL
- **index**: The frame/data point number being processed
- **uri**: The dataset-specific part of the URL
- **root_uri**: The base URL of the Tiled server
- **live_indices**: Array of all indices that have been processed
- **cum_size**: The maximum index seen + 1, used to track how many data points have been processed
- **data_project_dict**: Dictionary that stores metadata about the datasets being processed

## Potential Issues

1. If `live_indices` is empty, `Math.max(...live_indices)` will result in `-Infinity`, causing issues with `cum_size`
2. When models are updated, `live_indices` is reset but `data_project_dict` might not be, leading to inconsistent indices
3. The fallback to dataset matching (lines 34-41) could give incorrect indices if the URI matches but the index should be different

## Recommendations

1. Use a safer calculation for `cum_size`:
   ```javascript
   let cum_size = live_indices.length > 0 ? Math.max(...live_indices) + 1 : index + 1;
   ```

2. Always reset `data_project_dict` when models are updated:
   ```javascript
   reset_data_project_dict = {
       "root_uri": "",
       "data_type": "tiled",
       "datasets": [],
       "project_id": "live",
       "live_models": selected_models
   }
   ```

3. Consider using the index from the message directly rather than relying on dataset matching:
   ```javascript
   // Skip the fallback to dataset matching
   if (!sliceParam) {
       // Use the index from the message directly
       // No need to update it from datasets
   }
   ```
