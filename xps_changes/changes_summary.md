# XPS-LSE Integration Changes Summary

## Overview
This document summarizes the changes needed to integrate XPS with LSE using Tiled for data storage and WebSocket for notifications.

## Architecture
- **XPS**: Publishes `shot_mean` frames to Tiled
- **XPS**: Sends metadata (including `tiled_url`) via WebSocket
- **LSE**: Receives WebSocket notifications and reads data from Tiled

---

## Summary of Changes

### ArroyoXPS Repository (3 files modified):

#### 1. `src/tr_ap_xps/tiled.py` - Added `shot_mean` support to TiledPublisher
- Added `shot_mean: ArrayClient = None` field to `TiledScan` dataclass
- Added `patch_tiiled_frame()` call for `shot_mean` in `update_tiled_scan()`
- Added `shot_mean` array creation in `create_data_nodes()`

#### 2. `src/tr_ap_xps/websockets.py` - Include metadata in WebSocket messages
- Added `tiled_url` to start message payload
- Added `tiled_url`, `scan_name`, and `shot_num` to frame messages

#### 3. `src/tr_ap_xps/apps/processor_cli.py` - Enable Tiled publisher
- Uncommented `TiledPublisher` initialization
- Uncommented `operator.add_publisher(tiled_pub)`

### ArroyoSAS Repository (1 file modified):

#### 4. `XPSWebSocketListener` - Simplified to read from Tiled
- Removed Redis dependency and experiment name logic
- Extract `tiled_url` and `scan_name` from WebSocket messages
- Construct Tiled path using received metadata instead of building internally

---

## Other Files to Change?

**No other files need to be changed** if:
- The existing `lse_operator` in ArroyoSAS already uses the standard Arroyo operator pattern
- The `RawFrameEvent` schema already exists in `arroyosas.schemas`

**However, you may need to verify:**

### 1. Settings/Configuration files
Ensure the WebSocket URL is correctly configured in both repositories:
- **ArroyoXPS**: `settings.yaml` or `settings_container.yaml` should have `websocket_url` defined
- **ArroyoSAS**: Configuration should point to XPS WebSocket server (e.g., `ws://processor:8001/xps_operator`)

### 2. Docker compose
If using containers, ensure network connectivity between XPS processor and LSE services

---

## Data Flow

1. **XPS receives frame from LabView** → Processes frame → Computes `shot_mean`
2. **XPS writes to Tiled** → `{tiled_url}/runs/{scan_name}/shot_mean`
3. **XPS sends WebSocket notification** → Includes `tiled_url`, `scan_name`, `shot_num`
4. **LSE receives WebSocket** → Extracts metadata
5. **LSE reads from Tiled** → Uses provided `tiled_url` to fetch `shot_mean`

---

## Benefits of This Architecture

- **Decouples data transfer from notification**: WebSocket only sends metadata, not large arrays
- **Scalable**: Multiple consumers can read from Tiled without affecting XPS
- **Persistent**: Data remains in Tiled for later analysis
- **Simple**: Reuses existing Tiled infrastructure and Arroyo patterns

---

## Testing Checklist

- [ ] XPS writes `shot_mean` to Tiled successfully
- [ ] WebSocket messages include `tiled_url`, `scan_name`, `shot_num`
- [ ] LSE receives WebSocket notifications
- [ ] LSE can read `shot_mean` from Tiled using provided URL
- [ ] Data flow works end-to-end from LabView → XPS → Tiled → LSE

---

Generated: 2025-01-06
